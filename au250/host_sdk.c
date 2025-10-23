#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <sys/time.h> // for gettimeofday()

typedef unsigned int u32;
typedef unsigned long int u64;

#define EDGELLM_BASE_ADDR 0x00

#define VOCAB_SIZE 32000

// #define LLM_SIZE 4024909824

typedef struct
{
    char *str;
    int id;
} TokenIndex;

typedef struct
{
    char **vocab;
    float *vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

char *global_vocab[VOCAB_SIZE];
TokenIndex global_sort_vocab[VOCAB_SIZE];
float global_vocab_scores[VOCAB_SIZE];
char str_buffer[1024];
char vocab_space[177865 + 32000];
// int prompt_tokens[1024]={1,   518, 25580, 29962, 24948,   592,  1554,  1048, 12537, 14156, 518, 29914, 25580, 29962};
int prompt_tokens[1024];
int num_prompt_tokens = 14;

#define MAX_PROMPT_LEN 64
#define MAX_DECODE_LEN 900
#define NUM_TURN 1

char *tokenizer_path = "tkz.bin";
char prompt[MAX_PROMPT_LEN];
char rendered_prompt[512];

int compare_tokens(const void *a, const void *b);
void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer *t);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void sort_vocab(Tokenizer *t);
void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens);
char *decode(Tokenizer *t, int prev_token, int token);
void safe_printf(char *piece);

u32 Xil_In32(u64 addr);
void Xil_Out32(u64 addr, u32 data);

u32 token_list[2048];

int main()
{

    // u32 ret;
    // ret = Xil_In32(EDGELLM_BASE_ADDR + 0x0c);
    // printf("ret = %d\n", ret);
    // Xil_Out32(EDGELLM_BASE_ADDR + 0x0c, 21);
    // ret = Xil_In32(EDGELLM_BASE_ADDR + 0x0c);
    // printf("ret = %d\n", ret);
    // Xil_Out32(EDGELLM_BASE_ADDR + 0x0c, 24);
    // ret = Xil_In32(EDGELLM_BASE_ADDR + 0x0c);
    // printf("ret = %d\n", ret);
    // system("./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/data/llmc0.bin -s 2012725248  -a 0x000000000 -c 1");
    // system("./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/data/llmc1.bin -s 2012725248  -a 0x400000000 -c 1");

    u32 currTk;
    u32 decodeTk;

    printf("Init tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, VOCAB_SIZE);
    printf("Init Finish!\n");

    sort_vocab(&tokenizer);
    printf("Sort Finish!\n");

    struct timeval start_time, end_time;

    int token_cnt = 0;
    int prev_token = 0;
    for (int t = 0; t < NUM_TURN; t++)
    {
        // Xil_Out32(EDGELLM_BASE_ADDR+0xC0, 1);
        printf("Turn: %d, User Prompt:\n", t);

        fgets(prompt, MAX_PROMPT_LEN, stdin);
        prompt[strlen(prompt) - 1] = '\0';
        printf("%s\n", prompt);
        sprintf(rendered_prompt, "[INST] %s [/INST]", prompt);

        encode(&tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
        printf("Num of prefill token=%d, LLM Response:\n", num_prompt_tokens);

        u32 total_token = num_prompt_tokens + MAX_DECODE_LEN;
        Xil_Out32(EDGELLM_BASE_ADDR+0x10, total_token);
        token_cnt = 0;

        gettimeofday(&start_time, NULL);
        for (int i = 0; i < num_prompt_tokens - 1; i++)
        {
            token_list[token_cnt] = prompt_tokens[i];
            token_cnt++;
            Xil_Out32(EDGELLM_BASE_ADDR, 0x00030000 + prompt_tokens[i]);
        }
        token_list[token_cnt] = prompt_tokens[num_prompt_tokens - 1];
        token_cnt++;

        Xil_Out32(EDGELLM_BASE_ADDR, 0x00050000 + prompt_tokens[num_prompt_tokens - 1]);

        currTk = 0;
        prev_token = 0;

        for (int i = 0; i < MAX_DECODE_LEN; i++)
        {
            while (((currTk >> 31) & 0x01) != 1)
            {
                currTk = Xil_In32(EDGELLM_BASE_ADDR + 0x04);
            }
            decodeTk = (currTk >> 16) & 0x7fff;
            token_list[token_cnt] = decodeTk;
            token_cnt++;
            Xil_Out32(EDGELLM_BASE_ADDR, 0x00090000 + decodeTk);
            Xil_Out32(EDGELLM_BASE_ADDR + 0x80, 1);
            currTk = 0;
            char *piece = decode(&tokenizer, prev_token, decodeTk);
            safe_printf(piece);
            if (decodeTk == 2)
            {
                break;
            }
            prev_token = decodeTk;
        }
        gettimeofday(&end_time, NULL);

        u32 timer = Xil_In32(EDGELLM_BASE_ADDR+0x10);
        double timer_time = 1e-9 * (double)(timer) * 3.333;
        double elapsed_time = (end_time.tv_sec - start_time.tv_sec) + (end_time.tv_usec - start_time.tv_usec) / 1000000.0;
        
        printf("\nElapsed time: %.6f seconds, token/s: %f \n", elapsed_time, token_cnt / elapsed_time);
        // printf("Timer time: %.6f seconds, token/s: %f \n", timer_time, total_token / timer_time);
        printf("Finish Decoding!\n");
    }
    for (int i = 0; i < 32; i++)
    {
        printf("%d ", token_list[i]);
    }
    printf("\n");

    free_tokenizer(&tokenizer);
    return 0;
}

void Xil_Out32(u64 addr, u32 data)
{
    char command[128]; // Adjust size as necessary
    int ret;

    // Construct the command string
    snprintf(command, sizeof(command), "./reg_rw /dev/xdma0_user 0x%lx w 0x%x > /dev/null 2>&1", addr, data);

    // Execute the command
    ret = system(command);

    // Check for errors in system call
    if (ret == -1)
    {
        perror("Error executing command");
    }
    else if (WEXITSTATUS(ret) != 0)
    {
        fprintf(stderr, "Command exited with non-zero status: %d\n", WEXITSTATUS(ret));
    }
}

u32 Xil_In32(u64 addr)
{
    char command[256]; // Buffer for the command
    char buffer[256];  // Buffer for command output
    FILE *pipe;        // Pipe for capturing command output
    u32 result = 0;    // Variable to store the extracted result

    // Construct the command string
    snprintf(command, sizeof(command), "./reg_rw /dev/xdma0_user 0x%lx r", addr);

    // Open a pipe to the command
    pipe = popen(command, "r");
    if (!pipe)
    {
        perror("popen failed");
        return 0;
    }

    // Read the command output line by line
    while (fgets(buffer, sizeof(buffer), pipe) != NULL)
    {
        // Look for the line containing the read value
        if (strstr(buffer, "Read 32-bit value") != NULL)
        {
            // Extract the value after the colon and parse it as hexadecimal
            char *value_str = strrchr(buffer, ':');
            if (value_str)
            {
                sscanf(value_str + 1, " 0x%x", &result);
            }
        }
    }

    // Close the pipe and check for errors
    if (pclose(pipe) == -1)
    {
        perror("pclose failed");
    }

    return result; // Return the extracted value
}

int compare_tokens(const void *a, const void *b)
{
    return strcmp(((TokenIndex *)a)->str, ((TokenIndex *)b)->str);
}

void build_tokenizer(Tokenizer *t, char *tokenizer_path, int vocab_size)
{

    t->vocab_size = vocab_size;
    t->vocab = global_vocab;
    t->vocab_scores = global_vocab_scores;
    t->sorted_vocab = global_sort_vocab;

    for (int i = 0; i < 256; i++)
    {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }
    // read in the file
    FILE *file = fopen(tokenizer_path, "rb");
    if (!file)
    {
        fprintf(stderr, "couldn't load %s\n", tokenizer_path);
        exit(EXIT_FAILURE);
    }
    if (fread(&t->max_token_length, sizeof(int), 1, file) != 1)
    {
        fprintf(stderr, "failed read\n");
        exit(EXIT_FAILURE);
    }
    int len;
    for (int i = 0; i < vocab_size; i++)
    {
        if (fread(t->vocab_scores + i, sizeof(float), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        if (fread(&len, sizeof(int), 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i] = (char *)malloc(len + 1);
        if (fread(t->vocab[i], len, 1, file) != 1)
        {
            fprintf(stderr, "failed read\n");
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // add the string terminating token
    }
    fclose(file);
}

void free_tokenizer(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        free(t->vocab[i]);
    }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size)
{
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = {.str = str}; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void sort_vocab(Tokenizer *t)
{
    for (int i = 0; i < t->vocab_size; i++)
    {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void encode(Tokenizer *t, char *text, int bos, int eos, int *tokens, int *n_tokens)
{
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL)
    {
        printf("cannot encode NULL text\n");
        exit(EXIT_FAILURE);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos)
        tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0')
    {
        int dummy_prefix = str_lookup(" ", t->sorted_vocab, t->vocab_size);
        tokens[(*n_tokens)++] = dummy_prefix;
    }

    // Okay UTF-8 time. This will get messy. Here is the reference from Wikipedia:
    // Code point ↔ UTF-8 conversion
    // First code point	Last code point	Byte 1	Byte 2	Byte 3	Byte 4
    // U+0000	U+007F	    0xxxxxxx
    // U+0080	U+07FF	    110xxxxx	10xxxxxx
    // U+0800	U+FFFF	    1110xxxx	10xxxxxx	10xxxxxx
    // U+10000	U+10FFFF    11110xxx	10xxxxxx	10xxxxxx	10xxxxxx

    // process the raw (UTF-8) byte sequence of the input string
    for (char *c = text; *c != '\0'; c++)
    {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80)
        {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c + 1) & 0xC0) == 0x80 && str_len < 4)
        {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1)
        {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        }
        else
        {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i = 0; i < str_len; i++)
            {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1)
    {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i = 0; i < (*n_tokens - 1); i++)
        {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i + 1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score)
            {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1)
        {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx + 1; i < (*n_tokens - 1); i++)
        {
            tokens[i] = tokens[i + 1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos)
        tokens[(*n_tokens)++] = 2;
}

char *decode(Tokenizer *t, int prev_token, int token)
{
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ')
    {
        piece++;
    }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1)
    {
        piece = (char *)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece)
{
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL)
    {
        return;
    }
    if (piece[0] == '\0')
    {
        return;
    }
    if (piece[1] == '\0')
    {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val)))
        {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
    fflush(stdout);
}

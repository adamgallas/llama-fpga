#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include "platform.h"
#include "xil_printf.h"
#include "xil_types.h"
#include "xil_io.h"
#include "ff.h"
#include "xtime_l.h"
#include "xil_cache.h"
#include "sleep.h"


#define EDGELLM_BASE_ADDR 0x500000000

u64 *region_1=0x400000000;

FIL fil;
FATFS fatfs;
FRESULT response;

u32 wr_tot;

XTime tEnd, tCur;
u64 tUsed;

#define VOCAB_SIZE 32000

#define LLM_SIZE 4024909824

typedef struct {
    char *str;
    int id;
} TokenIndex;

typedef struct {
    char** vocab;
    float* vocab_scores;
    TokenIndex *sorted_vocab;
    int vocab_size;
    unsigned int max_token_length;
    unsigned char byte_pieces[512]; // stores all single-byte strings
} Tokenizer;

char *global_vocab[VOCAB_SIZE];
TokenIndex global_sort_vocab[VOCAB_SIZE];
float global_vocab_scores[VOCAB_SIZE];
char str_buffer[1024];
char vocab_space[177865+32000];
//int prompt_tokens[1024]={1,   518, 25580, 29962, 24948,   592,  1554,  1048, 12537, 14156, 518, 29914, 25580, 29962};
int prompt_tokens[1024];
int num_prompt_tokens = 14;

#define MAX_PROMPT_LEN 64
#define MAX_DECODE_LEN 1000
#define NUM_TURN 16

char *tokenizer_path = "tkz.bin";
char prompt[MAX_PROMPT_LEN];
char rendered_prompt[512];

int compare_tokens(const void *a, const void *b);
void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size);
void free_tokenizer(Tokenizer* t);
int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size);
void sort_vocab(Tokenizer* t);
void encode(Tokenizer* t, char *text, int bos, int eos, int *tokens, int *n_tokens);
char* decode(Tokenizer* t, int prev_token, int token);
void safe_printf(char *piece);

int main()
{
    init_platform();
	u32 currTk;
	u32 decodeTk;

    response = f_mount(&fatfs, "0:", 0);
    if(response != FR_OK){
    	printf("FATFS Mount Failed!\n");
    	return 0;
    }

    printf("Loading LLama2-7B...\n");
    XTime_GetTime(&tCur);
    response = f_open(&fil, "migllm.bin", FA_OPEN_EXISTING|FA_READ);
    printf("%d ", response);
    response = f_read(&fil, region_1, LLM_SIZE, &wr_tot);
    printf("%d ", response);
    response = f_close(&fil);
    printf("%d\n", response);
    XTime_GetTime(&tEnd);
    tUsed = ((tEnd - tCur) * 1000000) / (COUNTS_PER_SECOND);
    printf("Finish First Bank, %ld %ld\n", LLM_SIZE, wr_tot);
    printf("Elapsed %ld us\n", tUsed);

    printf("Load Finish!\n");

    Xil_DCacheFlush();

    printf("Init tokenizer...\n");
    Tokenizer tokenizer;
    build_tokenizer(&tokenizer, tokenizer_path, VOCAB_SIZE);
    printf("Init Finish!\n");

    sort_vocab(&tokenizer);
    printf("Sort Finish!\n");

    int token_cnt = 0;
    int prev_token = 0;
    for(int t=0;t<NUM_TURN;t++){
    	Xil_Out32(EDGELLM_BASE_ADDR+0xC0, 1);
    	sleep(1);
//		ret = Xil_In32(EDGELLM_BASE_ADDR+0x0c);
//		printf("%d\n", ret);
//		Xil_Out32(EDGELLM_BASE_ADDR+0x0c, 77);
//		ret = Xil_In32(EDGELLM_BASE_ADDR+0x0c);
//		printf("%d\n", ret);
		printf("Turn: %d, User Prompt:\n", t);

		fgets(prompt, MAX_PROMPT_LEN, stdin);
		prompt[strlen(prompt) - 1]='\0';
		printf("%s\n", prompt);
		sprintf(rendered_prompt, "[INST] %s [/INST]", prompt);

		XTime_GetTime(&tCur);
		encode(&tokenizer, rendered_prompt, 1, 0, prompt_tokens, &num_prompt_tokens);
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd - tCur) * 1000000) / (COUNTS_PER_SECOND);
		printf("Encoding Process Elapsed %ld us\n", tUsed);
		printf("LLM Response:\n");

		token_cnt = 0;
		XTime_GetTime(&tCur);
		for(int i=0;i<num_prompt_tokens-1;i++){
			token_cnt++;
			Xil_Out32(EDGELLM_BASE_ADDR, 0x00030000+prompt_tokens[i]);
		}
		token_cnt++;
		Xil_Out32(EDGELLM_BASE_ADDR, 0x00050000+prompt_tokens[num_prompt_tokens-1]);

		currTk = 0;
		prev_token = 0;
		for(int i=0;i<MAX_DECODE_LEN;i++){
			while(((currTk>>31) & 0x01) != 1){
				currTk = Xil_In32(EDGELLM_BASE_ADDR+0x04);
			}
			decodeTk = (currTk>>16) & 0x7fff;
			token_cnt++;
			Xil_Out32(EDGELLM_BASE_ADDR, 0x00090000+decodeTk);
			Xil_Out32(EDGELLM_BASE_ADDR+0x80, 1);
			currTk = 0;
			char *piece = decode(&tokenizer, prev_token, decodeTk);
			safe_printf(piece);
			if(decodeTk == 2){
				break;
			}
			prev_token = decodeTk;
		}
		XTime_GetTime(&tEnd);
		tUsed = ((tEnd - tCur) * 1000000) / (COUNTS_PER_SECOND);

		float ps_tps = 1000000.0 / ((float)(tUsed) / token_cnt);
		printf("\nElapsed PS %ld us, tps %f, total token: %d\n", tUsed, ps_tps, token_cnt);
		printf("Finish Decoding!\n");
		sleep(1);
    }

    free_tokenizer(&tokenizer);
    cleanup_platform();
    return 0;
}


int compare_tokens(const void *a, const void *b) {
    return strcmp(((TokenIndex*)a)->str, ((TokenIndex*)b)->str);
}

void build_tokenizer(Tokenizer* t, char* tokenizer_path, int vocab_size) {
    t->vocab_size = vocab_size;
    t->vocab = global_vocab;
    t->vocab_scores = global_vocab_scores;
    t->sorted_vocab = global_sort_vocab;

    for (int i = 0; i < 256; i++) {
        t->byte_pieces[i * 2] = (unsigned char)i;
        t->byte_pieces[i * 2 + 1] = '\0';
    }

    // FATFS structures and variables
    UINT read_tot;         // Variable to store the actual bytes read
    UINT current_offset = 0; // Track the current position in the file

    // Open the file with read access
    response = f_open(&fil, tokenizer_path, FA_OPEN_EXISTING | FA_READ);
    if (response != FR_OK) {
        printf("Couldn't open file %s, error code: %d\n", tokenizer_path, response);
        exit(EXIT_FAILURE);
    }

    // Read max_token_length
    response = f_read(&fil, &t->max_token_length, sizeof(int), &read_tot);
    if (response != FR_OK || read_tot != sizeof(int)) {
        printf("Failed to read max_token_length\n");
        f_close(&fil);
        exit(EXIT_FAILURE);
    }
    current_offset += sizeof(int); // Move file pointer forward by size of int

    int len;
    int vocab_space_index = 0;
    for (int i = 0; i < vocab_size; i++) {

        // Move file pointer to current offset before each read
        response = f_lseek(&fil, current_offset);
        if (response != FR_OK) {
            printf("Failed to seek to vocab_scores offset\n");
            f_close(&fil);
            exit(EXIT_FAILURE);
        }

        // Read vocab_scores
        response = f_read(&fil, t->vocab_scores + i, sizeof(float), &read_tot);
        if (response != FR_OK || read_tot != sizeof(float)) {
            printf("Failed to read vocab_scores\n");
            f_close(&fil);
            exit(EXIT_FAILURE);
        }
        current_offset += sizeof(float); // Update offset

        // Move file pointer to current offset before each read
        response = f_lseek(&fil, current_offset);
        if (response != FR_OK) {
            printf("Failed to seek to vocab length offset\n");
            f_close(&fil);
            exit(EXIT_FAILURE);
        }

        // Read length of the vocab word
        response = f_read(&fil, &len, sizeof(int), &read_tot);
        if (response != FR_OK || read_tot != sizeof(int)) {
            printf("Failed to read vocab length\n");
            f_close(&fil);
            exit(EXIT_FAILURE);
        }
        current_offset += sizeof(int); // Update offset

        // Allocate memory for vocab word and read the word itself
        // t->vocab[i] = (char *)malloc(len + 1);

        t->vocab[i] = vocab_space + vocab_space_index;
        vocab_space_index += len + 1;

        if (!t->vocab[i]) {
            printf("Memory allocation failed for vocab %d\n", i);
            f_close(&fil);
            exit(EXIT_FAILURE);
        }

        // Move file pointer to current offset before each read
        response = f_lseek(&fil, current_offset);
        if (response != FR_OK) {
            printf("Failed to seek to vocab word offset\n");
            f_close(&fil);
            exit(EXIT_FAILURE);
        }

        // Read the actual vocab word
        response = f_read(&fil, t->vocab[i], len, &read_tot);
        if (response != FR_OK || read_tot != len) {
            printf("Failed to read vocab word\n");
            free(t->vocab[i]);
            f_close(&fil);
            exit(EXIT_FAILURE);
        }
        t->vocab[i][len] = '\0'; // Add the string terminating character
        current_offset += len; // Update offset
    }

    // Close the file
    response = f_close(&fil);
    if (response != FR_OK) {
        printf("Failed to close the file, error code: %d\n", response);
        exit(EXIT_FAILURE);
    }
}


void free_tokenizer(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) { free(t->vocab[i]); }
}

int str_lookup(char *str, TokenIndex *sorted_vocab, int vocab_size) {
    // efficiently find the perfect match for str in vocab, return its index or -1 if not found
    TokenIndex tok = { .str = str }; // acts as the key to search for
    TokenIndex *res = bsearch(&tok, sorted_vocab, vocab_size, sizeof(TokenIndex), compare_tokens);
    return res != NULL ? res->id : -1;
}

void sort_vocab(Tokenizer* t) {
    for (int i = 0; i < t->vocab_size; i++) {
        t->sorted_vocab[i].str = t->vocab[i];
        t->sorted_vocab[i].id = i;
    }
    qsort(t->sorted_vocab, t->vocab_size, sizeof(TokenIndex), compare_tokens);
}

void encode(Tokenizer* t, char *text, int bos, int eos, int *tokens, int *n_tokens) {
    // encode the string text (input) into an upper-bound preallocated tokens[] array
    // bos != 0 means prepend the BOS token (=1), eos != 0 means append the EOS token (=2)
    if (text == NULL) {
    	printf("cannot encode NULL text\n");
    	exit(EXIT_FAILURE);
    }

    // create a temporary buffer that will store merge candidates of always two consecutive tokens
    // *2 for concat, +1 for null terminator +2 for UTF8 (in case max_token_length is 1)
    size_t str_len = 0;

    // start at 0 tokens
    *n_tokens = 0;

    // add optional BOS (=1) token, if desired
    if (bos) tokens[(*n_tokens)++] = 1;

    // add_dummy_prefix is true by default
    // so prepend a dummy prefix token to the input string, but only if text != ""
    // TODO: pretty sure this isn't correct in the general case but I don't have the
    // energy to read more of the sentencepiece code to figure out what it's doing
    if (text[0] != '\0') {
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
    for (char *c = text; *c != '\0'; c++) {

        // reset buffer if the current byte is ASCII or a leading byte
        // 0xC0 is 11000000, so (*c & 0xC0) keeps the first 2 bits and zeros the rest
        // 0x80 is 10000000
        // in UTF-8, all continuation bytes start with "10" in first two bits
        // so in English this is: "if this byte is not a continuation byte"
        if ((*c & 0xC0) != 0x80) {
            // this byte must be either a leading byte (11...) or an ASCII char (0x...)
            // => reset our location, as we're starting a new UTF-8 codepoint
            str_len = 0;
        }

        // append the current byte to the buffer
        str_buffer[str_len++] = *c; // ++ is post-increment, incremented after this line
        str_buffer[str_len] = '\0';

        // while the next character is a continuation byte, continue appending
        // but if there are too many of them, just stop to avoid overruning str_buffer size.
        if ((*(c+1) & 0xC0) == 0x80 && str_len < 4) {
            continue;
        }

        // ok c+1 is not a continuation byte, so we've read in a full codepoint
        int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);

        if (id != -1) {
            // we found this codepoint in vocab, add it as a token
            tokens[(*n_tokens)++] = id;
        } else {
            // byte_fallback encoding: just encode each byte as a token
            // +3 is here because the first 3 vocab elements are <unk>, <s>, </s>
            // so the individual bytes only start at index 3
            for (int i=0; i < str_len; i++) {
                tokens[(*n_tokens)++] = (unsigned char)str_buffer[i] + 3;
            }
        }
        str_len = 0; // protect against a sequence of stray UTF8 continuation bytes
    }

    // merge the best consecutive pair each iteration, according the scores in vocab_scores
    while (1) {
        float best_score = -1e10;
        int best_id = -1;
        int best_idx = -1;

        for (int i=0; i < (*n_tokens-1); i++) {
            // check if we can merge the pair (tokens[i], tokens[i+1])
            sprintf(str_buffer, "%s%s", t->vocab[tokens[i]], t->vocab[tokens[i+1]]);
            int id = str_lookup(str_buffer, t->sorted_vocab, t->vocab_size);
            if (id != -1 && t->vocab_scores[id] > best_score) {
                // this merge pair exists in vocab! record its score and position
                best_score = t->vocab_scores[id];
                best_id = id;
                best_idx = i;
            }
        }

        if (best_idx == -1) {
            break; // we couldn't find any more pairs to merge, so we're done
        }

        // merge the consecutive pair (best_idx, best_idx+1) into new token best_id
        tokens[best_idx] = best_id;
        // delete token at position best_idx+1, shift the entire sequence back 1
        for (int i = best_idx+1; i < (*n_tokens-1); i++) {
            tokens[i] = tokens[i+1];
        }
        (*n_tokens)--; // token length decreased
    }

    // add optional EOS (=2) token, if desired
    if (eos) tokens[(*n_tokens)++] = 2;
}

char* decode(Tokenizer* t, int prev_token, int token) {
    char *piece = t->vocab[token];
    // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
    if (prev_token == 1 && piece[0] == ' ') { piece++; }
    // careful, some tokens designate raw bytes, and look like e.g. '<0x01>'
    // parse this and convert and return the actual byte
    unsigned char byte_val;
    if (sscanf(piece, "<0x%02hhX>", &byte_val) == 1) {
        piece = (char*)t->byte_pieces + byte_val * 2;
    }
    return piece;
}

void safe_printf(char *piece) {
    // piece might be a raw byte token, and we only want to print printable chars or whitespace
    // because some of the other bytes can be various control codes, backspace, etc.
    if (piece == NULL) { return; }
    if (piece[0] == '\0') { return; }
    if (piece[1] == '\0') {
        unsigned char byte_val = piece[0];
        if (!(isprint(byte_val) || isspace(byte_val))) {
            return; // bad byte, don't print it
        }
    }
    printf("%s", piece);
    fflush(stdout);
}

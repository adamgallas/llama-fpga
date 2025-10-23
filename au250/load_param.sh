./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/llama2-7b-s4/llmc0.bin -s 1043070976 -a 0x000000000 -c 1

./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/llama2-7b-s4/llmc1.bin -s 1043070976 -a 0x400000000 -c 1

./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/llama2-7b-s4/llmc2.bin -s 1043070976 -a 0x800000000 -c 1

./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/llama2-7b-s4/llmc3.bin -s 1043070976 -a 0xc00000000 -c 1

# sudo ./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/data/migllm.bin -s 4024909824  -a 0x000000000 -c 1

# sudo ./dma_to_device -d /dev/xdma0_h2c_0 -f ../tests/data/alveoTest.bin -s 4024909824  -a 0x000000000 -c 1

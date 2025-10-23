--Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
----------------------------------------------------------------------------------
--Command: generate_target bd_a818_wrapper.bd
--Design : bd_a818_wrapper
--Purpose: IP block netlist
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity bd_a818_wrapper is
  port (
    Clk : in STD_LOGIC;
    IO_addr_strobe : out STD_LOGIC;
    IO_address : out STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_byte_enable : out STD_LOGIC_VECTOR ( 3 downto 0 );
    IO_read_data : in STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_read_strobe : out STD_LOGIC;
    IO_ready : in STD_LOGIC;
    IO_write_data : out STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_write_strobe : out STD_LOGIC;
    Reset : in STD_LOGIC;
    TRACE_data_access : out STD_LOGIC;
    TRACE_data_address : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_data_byte_enable : out STD_LOGIC_VECTOR ( 0 to 3 );
    TRACE_data_read : out STD_LOGIC;
    TRACE_data_write : out STD_LOGIC;
    TRACE_data_write_value : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_dcache_hit : out STD_LOGIC;
    TRACE_dcache_rdy : out STD_LOGIC;
    TRACE_dcache_read : out STD_LOGIC;
    TRACE_dcache_req : out STD_LOGIC;
    TRACE_delay_slot : out STD_LOGIC;
    TRACE_ex_piperun : out STD_LOGIC;
    TRACE_exception_kind : out STD_LOGIC_VECTOR ( 0 to 4 );
    TRACE_exception_taken : out STD_LOGIC;
    TRACE_icache_hit : out STD_LOGIC;
    TRACE_icache_rdy : out STD_LOGIC;
    TRACE_icache_req : out STD_LOGIC;
    TRACE_instruction : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_jump_hit : out STD_LOGIC;
    TRACE_jump_taken : out STD_LOGIC;
    TRACE_mb_halted : out STD_LOGIC;
    TRACE_mem_piperun : out STD_LOGIC;
    TRACE_msr_reg : out STD_LOGIC_VECTOR ( 0 to 14 );
    TRACE_new_reg_value : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_of_piperun : out STD_LOGIC;
    TRACE_pc : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_pid_reg : out STD_LOGIC_VECTOR ( 0 to 7 );
    TRACE_reg_addr : out STD_LOGIC_VECTOR ( 0 to 4 );
    TRACE_reg_write : out STD_LOGIC;
    TRACE_valid_instr : out STD_LOGIC
  );
end bd_a818_wrapper;

architecture STRUCTURE of bd_a818_wrapper is
  component bd_a818 is
  port (
    Clk : in STD_LOGIC;
    Reset : in STD_LOGIC;
    TRACE_data_access : out STD_LOGIC;
    TRACE_data_address : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_data_byte_enable : out STD_LOGIC_VECTOR ( 0 to 3 );
    TRACE_data_read : out STD_LOGIC;
    TRACE_data_write : out STD_LOGIC;
    TRACE_data_write_value : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_dcache_hit : out STD_LOGIC;
    TRACE_dcache_rdy : out STD_LOGIC;
    TRACE_dcache_read : out STD_LOGIC;
    TRACE_dcache_req : out STD_LOGIC;
    TRACE_delay_slot : out STD_LOGIC;
    TRACE_ex_piperun : out STD_LOGIC;
    TRACE_exception_kind : out STD_LOGIC_VECTOR ( 0 to 4 );
    TRACE_exception_taken : out STD_LOGIC;
    TRACE_icache_hit : out STD_LOGIC;
    TRACE_icache_rdy : out STD_LOGIC;
    TRACE_icache_req : out STD_LOGIC;
    TRACE_instruction : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_jump_hit : out STD_LOGIC;
    TRACE_jump_taken : out STD_LOGIC;
    TRACE_mb_halted : out STD_LOGIC;
    TRACE_mem_piperun : out STD_LOGIC;
    TRACE_msr_reg : out STD_LOGIC_VECTOR ( 0 to 14 );
    TRACE_new_reg_value : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_of_piperun : out STD_LOGIC;
    TRACE_pc : out STD_LOGIC_VECTOR ( 0 to 31 );
    TRACE_pid_reg : out STD_LOGIC_VECTOR ( 0 to 7 );
    TRACE_reg_addr : out STD_LOGIC_VECTOR ( 0 to 4 );
    TRACE_reg_write : out STD_LOGIC;
    TRACE_valid_instr : out STD_LOGIC;
    IO_addr_strobe : out STD_LOGIC;
    IO_address : out STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_byte_enable : out STD_LOGIC_VECTOR ( 3 downto 0 );
    IO_read_data : in STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_read_strobe : out STD_LOGIC;
    IO_ready : in STD_LOGIC;
    IO_write_data : out STD_LOGIC_VECTOR ( 31 downto 0 );
    IO_write_strobe : out STD_LOGIC
  );
  end component bd_a818;
begin
bd_a818_i: component bd_a818
     port map (
      Clk => Clk,
      IO_addr_strobe => IO_addr_strobe,
      IO_address(31 downto 0) => IO_address(31 downto 0),
      IO_byte_enable(3 downto 0) => IO_byte_enable(3 downto 0),
      IO_read_data(31 downto 0) => IO_read_data(31 downto 0),
      IO_read_strobe => IO_read_strobe,
      IO_ready => IO_ready,
      IO_write_data(31 downto 0) => IO_write_data(31 downto 0),
      IO_write_strobe => IO_write_strobe,
      Reset => Reset,
      TRACE_data_access => TRACE_data_access,
      TRACE_data_address(0 to 31) => TRACE_data_address(0 to 31),
      TRACE_data_byte_enable(0 to 3) => TRACE_data_byte_enable(0 to 3),
      TRACE_data_read => TRACE_data_read,
      TRACE_data_write => TRACE_data_write,
      TRACE_data_write_value(0 to 31) => TRACE_data_write_value(0 to 31),
      TRACE_dcache_hit => TRACE_dcache_hit,
      TRACE_dcache_rdy => TRACE_dcache_rdy,
      TRACE_dcache_read => TRACE_dcache_read,
      TRACE_dcache_req => TRACE_dcache_req,
      TRACE_delay_slot => TRACE_delay_slot,
      TRACE_ex_piperun => TRACE_ex_piperun,
      TRACE_exception_kind(0 to 4) => TRACE_exception_kind(0 to 4),
      TRACE_exception_taken => TRACE_exception_taken,
      TRACE_icache_hit => TRACE_icache_hit,
      TRACE_icache_rdy => TRACE_icache_rdy,
      TRACE_icache_req => TRACE_icache_req,
      TRACE_instruction(0 to 31) => TRACE_instruction(0 to 31),
      TRACE_jump_hit => TRACE_jump_hit,
      TRACE_jump_taken => TRACE_jump_taken,
      TRACE_mb_halted => TRACE_mb_halted,
      TRACE_mem_piperun => TRACE_mem_piperun,
      TRACE_msr_reg(0 to 14) => TRACE_msr_reg(0 to 14),
      TRACE_new_reg_value(0 to 31) => TRACE_new_reg_value(0 to 31),
      TRACE_of_piperun => TRACE_of_piperun,
      TRACE_pc(0 to 31) => TRACE_pc(0 to 31),
      TRACE_pid_reg(0 to 7) => TRACE_pid_reg(0 to 7),
      TRACE_reg_addr(0 to 4) => TRACE_reg_addr(0 to 4),
      TRACE_reg_write => TRACE_reg_write,
      TRACE_valid_instr => TRACE_valid_instr
    );
end STRUCTURE;

--Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
----------------------------------------------------------------------------------
--Tool Version: Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
--Date        : Fri Feb 14 15:16:39 2025
--Host        : adam running 64-bit Ubuntu 22.04.2 LTS
--Command     : generate_target kv260bd_wrapper.bd
--Design      : kv260bd_wrapper
--Purpose     : IP block netlist
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity kv260bd_wrapper is
end kv260bd_wrapper;

architecture STRUCTURE of kv260bd_wrapper is
  component kv260bd is
  end component kv260bd;
begin
kv260bd_i: component kv260bd
 ;
end STRUCTURE;

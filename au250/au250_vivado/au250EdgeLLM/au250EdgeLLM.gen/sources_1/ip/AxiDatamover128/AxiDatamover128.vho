-- (c) Copyright 1995-2024 Xilinx, Inc. All rights reserved.
-- 
-- This file contains confidential and proprietary information
-- of Xilinx, Inc. and is protected under U.S. and
-- international copyright and other intellectual property
-- laws.
-- 
-- DISCLAIMER
-- This disclaimer is not a license and does not grant any
-- rights to the materials distributed herewith. Except as
-- otherwise provided in a valid license issued to you by
-- Xilinx, and to the maximum extent permitted by applicable
-- law: (1) THESE MATERIALS ARE MADE AVAILABLE "AS IS" AND
-- WITH ALL FAULTS, AND XILINX HEREBY DISCLAIMS ALL WARRANTIES
-- AND CONDITIONS, EXPRESS, IMPLIED, OR STATUTORY, INCLUDING
-- BUT NOT LIMITED TO WARRANTIES OF MERCHANTABILITY, NON-
-- INFRINGEMENT, OR FITNESS FOR ANY PARTICULAR PURPOSE; and
-- (2) Xilinx shall not be liable (whether in contract or tort,
-- including negligence, or under any other theory of
-- liability) for any loss or damage of any kind or nature
-- related to, arising under or in connection with these
-- materials, including for any direct, or any indirect,
-- special, incidental, or consequential loss or damage
-- (including loss of data, profits, goodwill, or any type of
-- loss or damage suffered as a result of any action brought
-- by a third party) even if such damage or loss was
-- reasonably foreseeable or Xilinx had been advised of the
-- possibility of the same.
-- 
-- CRITICAL APPLICATIONS
-- Xilinx products are not designed or intended to be fail-
-- safe, or for use in any application requiring fail-safe
-- performance, such as life-support or safety devices or
-- systems, Class III medical devices, nuclear facilities,
-- applications related to the deployment of airbags, or any
-- other applications that could lead to death, personal
-- injury, or severe property or environmental damage
-- (individually and collectively, "Critical
-- Applications"). Customer assumes the sole risk and
-- liability of any use of Xilinx products in Critical
-- Applications, subject only to applicable laws and
-- regulations governing limitations on product liability.
-- 
-- THIS COPYRIGHT NOTICE AND DISCLAIMER MUST BE RETAINED AS
-- PART OF THIS FILE AT ALL TIMES.
-- 
-- DO NOT MODIFY THIS FILE.
-- IP VLNV: xilinx.com:ip:axi_datamover:5.1
-- IP Revision: 29

-- The following code must appear in the VHDL architecture header.

------------- Begin Cut here for COMPONENT Declaration ------ COMP_TAG
COMPONENT AxiDatamover128
  PORT (
    m_axi_mm2s_aclk : IN STD_LOGIC;
    m_axi_mm2s_aresetn : IN STD_LOGIC;
    mm2s_err : OUT STD_LOGIC;
    m_axis_mm2s_cmdsts_aclk : IN STD_LOGIC;
    m_axis_mm2s_cmdsts_aresetn : IN STD_LOGIC;
    s_axis_mm2s_cmd_tvalid : IN STD_LOGIC;
    s_axis_mm2s_cmd_tready : OUT STD_LOGIC;
    s_axis_mm2s_cmd_tdata : IN STD_LOGIC_VECTOR(71 DOWNTO 0);
    m_axis_mm2s_sts_tvalid : OUT STD_LOGIC;
    m_axis_mm2s_sts_tready : IN STD_LOGIC;
    m_axis_mm2s_sts_tdata : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    m_axis_mm2s_sts_tkeep : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    m_axis_mm2s_sts_tlast : OUT STD_LOGIC;
    m_axi_mm2s_arid : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_mm2s_araddr : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    m_axi_mm2s_arlen : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    m_axi_mm2s_arsize : OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    m_axi_mm2s_arburst : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    m_axi_mm2s_arprot : OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    m_axi_mm2s_arcache : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_mm2s_aruser : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_mm2s_arvalid : OUT STD_LOGIC;
    m_axi_mm2s_arready : IN STD_LOGIC;
    m_axi_mm2s_rdata : IN STD_LOGIC_VECTOR(127 DOWNTO 0);
    m_axi_mm2s_rresp : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    m_axi_mm2s_rlast : IN STD_LOGIC;
    m_axi_mm2s_rvalid : IN STD_LOGIC;
    m_axi_mm2s_rready : OUT STD_LOGIC;
    m_axis_mm2s_tdata : OUT STD_LOGIC_VECTOR(127 DOWNTO 0);
    m_axis_mm2s_tkeep : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    m_axis_mm2s_tlast : OUT STD_LOGIC;
    m_axis_mm2s_tvalid : OUT STD_LOGIC;
    m_axis_mm2s_tready : IN STD_LOGIC;
    m_axi_s2mm_aclk : IN STD_LOGIC;
    m_axi_s2mm_aresetn : IN STD_LOGIC;
    s2mm_err : OUT STD_LOGIC;
    m_axis_s2mm_cmdsts_awclk : IN STD_LOGIC;
    m_axis_s2mm_cmdsts_aresetn : IN STD_LOGIC;
    s_axis_s2mm_cmd_tvalid : IN STD_LOGIC;
    s_axis_s2mm_cmd_tready : OUT STD_LOGIC;
    s_axis_s2mm_cmd_tdata : IN STD_LOGIC_VECTOR(71 DOWNTO 0);
    m_axis_s2mm_sts_tvalid : OUT STD_LOGIC;
    m_axis_s2mm_sts_tready : IN STD_LOGIC;
    m_axis_s2mm_sts_tdata : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    m_axis_s2mm_sts_tkeep : OUT STD_LOGIC_VECTOR(0 DOWNTO 0);
    m_axis_s2mm_sts_tlast : OUT STD_LOGIC;
    m_axi_s2mm_awid : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_s2mm_awaddr : OUT STD_LOGIC_VECTOR(31 DOWNTO 0);
    m_axi_s2mm_awlen : OUT STD_LOGIC_VECTOR(7 DOWNTO 0);
    m_axi_s2mm_awsize : OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    m_axi_s2mm_awburst : OUT STD_LOGIC_VECTOR(1 DOWNTO 0);
    m_axi_s2mm_awprot : OUT STD_LOGIC_VECTOR(2 DOWNTO 0);
    m_axi_s2mm_awcache : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_s2mm_awuser : OUT STD_LOGIC_VECTOR(3 DOWNTO 0);
    m_axi_s2mm_awvalid : OUT STD_LOGIC;
    m_axi_s2mm_awready : IN STD_LOGIC;
    m_axi_s2mm_wdata : OUT STD_LOGIC_VECTOR(127 DOWNTO 0);
    m_axi_s2mm_wstrb : OUT STD_LOGIC_VECTOR(15 DOWNTO 0);
    m_axi_s2mm_wlast : OUT STD_LOGIC;
    m_axi_s2mm_wvalid : OUT STD_LOGIC;
    m_axi_s2mm_wready : IN STD_LOGIC;
    m_axi_s2mm_bresp : IN STD_LOGIC_VECTOR(1 DOWNTO 0);
    m_axi_s2mm_bvalid : IN STD_LOGIC;
    m_axi_s2mm_bready : OUT STD_LOGIC;
    s_axis_s2mm_tdata : IN STD_LOGIC_VECTOR(127 DOWNTO 0);
    s_axis_s2mm_tkeep : IN STD_LOGIC_VECTOR(15 DOWNTO 0);
    s_axis_s2mm_tlast : IN STD_LOGIC;
    s_axis_s2mm_tvalid : IN STD_LOGIC;
    s_axis_s2mm_tready : OUT STD_LOGIC 
  );
END COMPONENT;
-- COMP_TAG_END ------ End COMPONENT Declaration ------------

-- The following code must appear in the VHDL architecture
-- body. Substitute your own instance name and net names.

------------- Begin Cut here for INSTANTIATION Template ----- INST_TAG
your_instance_name : AxiDatamover128
  PORT MAP (
    m_axi_mm2s_aclk => m_axi_mm2s_aclk,
    m_axi_mm2s_aresetn => m_axi_mm2s_aresetn,
    mm2s_err => mm2s_err,
    m_axis_mm2s_cmdsts_aclk => m_axis_mm2s_cmdsts_aclk,
    m_axis_mm2s_cmdsts_aresetn => m_axis_mm2s_cmdsts_aresetn,
    s_axis_mm2s_cmd_tvalid => s_axis_mm2s_cmd_tvalid,
    s_axis_mm2s_cmd_tready => s_axis_mm2s_cmd_tready,
    s_axis_mm2s_cmd_tdata => s_axis_mm2s_cmd_tdata,
    m_axis_mm2s_sts_tvalid => m_axis_mm2s_sts_tvalid,
    m_axis_mm2s_sts_tready => m_axis_mm2s_sts_tready,
    m_axis_mm2s_sts_tdata => m_axis_mm2s_sts_tdata,
    m_axis_mm2s_sts_tkeep => m_axis_mm2s_sts_tkeep,
    m_axis_mm2s_sts_tlast => m_axis_mm2s_sts_tlast,
    m_axi_mm2s_arid => m_axi_mm2s_arid,
    m_axi_mm2s_araddr => m_axi_mm2s_araddr,
    m_axi_mm2s_arlen => m_axi_mm2s_arlen,
    m_axi_mm2s_arsize => m_axi_mm2s_arsize,
    m_axi_mm2s_arburst => m_axi_mm2s_arburst,
    m_axi_mm2s_arprot => m_axi_mm2s_arprot,
    m_axi_mm2s_arcache => m_axi_mm2s_arcache,
    m_axi_mm2s_aruser => m_axi_mm2s_aruser,
    m_axi_mm2s_arvalid => m_axi_mm2s_arvalid,
    m_axi_mm2s_arready => m_axi_mm2s_arready,
    m_axi_mm2s_rdata => m_axi_mm2s_rdata,
    m_axi_mm2s_rresp => m_axi_mm2s_rresp,
    m_axi_mm2s_rlast => m_axi_mm2s_rlast,
    m_axi_mm2s_rvalid => m_axi_mm2s_rvalid,
    m_axi_mm2s_rready => m_axi_mm2s_rready,
    m_axis_mm2s_tdata => m_axis_mm2s_tdata,
    m_axis_mm2s_tkeep => m_axis_mm2s_tkeep,
    m_axis_mm2s_tlast => m_axis_mm2s_tlast,
    m_axis_mm2s_tvalid => m_axis_mm2s_tvalid,
    m_axis_mm2s_tready => m_axis_mm2s_tready,
    m_axi_s2mm_aclk => m_axi_s2mm_aclk,
    m_axi_s2mm_aresetn => m_axi_s2mm_aresetn,
    s2mm_err => s2mm_err,
    m_axis_s2mm_cmdsts_awclk => m_axis_s2mm_cmdsts_awclk,
    m_axis_s2mm_cmdsts_aresetn => m_axis_s2mm_cmdsts_aresetn,
    s_axis_s2mm_cmd_tvalid => s_axis_s2mm_cmd_tvalid,
    s_axis_s2mm_cmd_tready => s_axis_s2mm_cmd_tready,
    s_axis_s2mm_cmd_tdata => s_axis_s2mm_cmd_tdata,
    m_axis_s2mm_sts_tvalid => m_axis_s2mm_sts_tvalid,
    m_axis_s2mm_sts_tready => m_axis_s2mm_sts_tready,
    m_axis_s2mm_sts_tdata => m_axis_s2mm_sts_tdata,
    m_axis_s2mm_sts_tkeep => m_axis_s2mm_sts_tkeep,
    m_axis_s2mm_sts_tlast => m_axis_s2mm_sts_tlast,
    m_axi_s2mm_awid => m_axi_s2mm_awid,
    m_axi_s2mm_awaddr => m_axi_s2mm_awaddr,
    m_axi_s2mm_awlen => m_axi_s2mm_awlen,
    m_axi_s2mm_awsize => m_axi_s2mm_awsize,
    m_axi_s2mm_awburst => m_axi_s2mm_awburst,
    m_axi_s2mm_awprot => m_axi_s2mm_awprot,
    m_axi_s2mm_awcache => m_axi_s2mm_awcache,
    m_axi_s2mm_awuser => m_axi_s2mm_awuser,
    m_axi_s2mm_awvalid => m_axi_s2mm_awvalid,
    m_axi_s2mm_awready => m_axi_s2mm_awready,
    m_axi_s2mm_wdata => m_axi_s2mm_wdata,
    m_axi_s2mm_wstrb => m_axi_s2mm_wstrb,
    m_axi_s2mm_wlast => m_axi_s2mm_wlast,
    m_axi_s2mm_wvalid => m_axi_s2mm_wvalid,
    m_axi_s2mm_wready => m_axi_s2mm_wready,
    m_axi_s2mm_bresp => m_axi_s2mm_bresp,
    m_axi_s2mm_bvalid => m_axi_s2mm_bvalid,
    m_axi_s2mm_bready => m_axi_s2mm_bready,
    s_axis_s2mm_tdata => s_axis_s2mm_tdata,
    s_axis_s2mm_tkeep => s_axis_s2mm_tkeep,
    s_axis_s2mm_tlast => s_axis_s2mm_tlast,
    s_axis_s2mm_tvalid => s_axis_s2mm_tvalid,
    s_axis_s2mm_tready => s_axis_s2mm_tready
  );
-- INST_TAG_END ------ End INSTANTIATION Template ---------

-- You must compile the wrapper file AxiDatamover128.vhd when simulating
-- the core, AxiDatamover128. When compiling the wrapper file, be sure to
-- reference the VHDL simulation library.




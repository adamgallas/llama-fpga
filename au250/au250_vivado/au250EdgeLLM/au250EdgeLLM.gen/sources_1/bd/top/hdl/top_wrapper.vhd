--Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
----------------------------------------------------------------------------------
--Tool Version: Vivado v.2022.2 (lin64) Build 3671981 Fri Oct 14 04:59:54 MDT 2022
--Date        : Sat Feb 22 20:40:28 2025
--Host        : adam running 64-bit Ubuntu 22.04.5 LTS
--Command     : generate_target top_wrapper.bd
--Design      : top_wrapper
--Purpose     : IP block netlist
----------------------------------------------------------------------------------
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;
library UNISIM;
use UNISIM.VCOMPONENTS.ALL;
entity top_wrapper is
  port (
    ddr4_sdram_c0_act_n : out STD_LOGIC;
    ddr4_sdram_c0_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c0_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c0_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c0_ck_c : out STD_LOGIC;
    ddr4_sdram_c0_ck_t : out STD_LOGIC;
    ddr4_sdram_c0_cke : out STD_LOGIC;
    ddr4_sdram_c0_cs_n : out STD_LOGIC;
    ddr4_sdram_c0_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c0_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c0_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c0_odt : out STD_LOGIC;
    ddr4_sdram_c0_par : out STD_LOGIC;
    ddr4_sdram_c0_reset_n : out STD_LOGIC;
    ddr4_sdram_c1_act_n : out STD_LOGIC;
    ddr4_sdram_c1_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c1_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c1_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c1_ck_c : out STD_LOGIC;
    ddr4_sdram_c1_ck_t : out STD_LOGIC;
    ddr4_sdram_c1_cke : out STD_LOGIC;
    ddr4_sdram_c1_cs_n : out STD_LOGIC;
    ddr4_sdram_c1_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c1_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c1_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c1_odt : out STD_LOGIC;
    ddr4_sdram_c1_par : out STD_LOGIC;
    ddr4_sdram_c1_reset_n : out STD_LOGIC;
    ddr4_sdram_c2_act_n : out STD_LOGIC;
    ddr4_sdram_c2_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c2_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c2_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c2_ck_c : out STD_LOGIC;
    ddr4_sdram_c2_ck_t : out STD_LOGIC;
    ddr4_sdram_c2_cke : out STD_LOGIC;
    ddr4_sdram_c2_cs_n : out STD_LOGIC;
    ddr4_sdram_c2_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c2_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c2_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c2_odt : out STD_LOGIC;
    ddr4_sdram_c2_par : out STD_LOGIC;
    ddr4_sdram_c2_reset_n : out STD_LOGIC;
    ddr4_sdram_c3_act_n : out STD_LOGIC;
    ddr4_sdram_c3_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c3_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c3_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c3_ck_c : out STD_LOGIC;
    ddr4_sdram_c3_ck_t : out STD_LOGIC;
    ddr4_sdram_c3_cke : out STD_LOGIC;
    ddr4_sdram_c3_cs_n : out STD_LOGIC;
    ddr4_sdram_c3_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c3_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c3_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c3_odt : out STD_LOGIC;
    ddr4_sdram_c3_par : out STD_LOGIC;
    ddr4_sdram_c3_reset_n : out STD_LOGIC;
    default_300mhz_clk0_clk_n : in STD_LOGIC;
    default_300mhz_clk0_clk_p : in STD_LOGIC;
    default_300mhz_clk1_clk_n : in STD_LOGIC;
    default_300mhz_clk1_clk_p : in STD_LOGIC;
    default_300mhz_clk2_clk_n : in STD_LOGIC;
    default_300mhz_clk2_clk_p : in STD_LOGIC;
    default_300mhz_clk3_clk_n : in STD_LOGIC;
    default_300mhz_clk3_clk_p : in STD_LOGIC;
    pci_express_x1_rxn : in STD_LOGIC;
    pci_express_x1_rxp : in STD_LOGIC;
    pci_express_x1_txn : out STD_LOGIC;
    pci_express_x1_txp : out STD_LOGIC;
    pcie_perstn : in STD_LOGIC;
    pcie_refclk_clk_n : in STD_LOGIC;
    pcie_refclk_clk_p : in STD_LOGIC;
    resetn : in STD_LOGIC
  );
end top_wrapper;

architecture STRUCTURE of top_wrapper is
  component top is
  port (
    pci_express_x1_rxn : in STD_LOGIC;
    pci_express_x1_rxp : in STD_LOGIC;
    pci_express_x1_txn : out STD_LOGIC;
    pci_express_x1_txp : out STD_LOGIC;
    pcie_refclk_clk_p : in STD_LOGIC;
    pcie_refclk_clk_n : in STD_LOGIC;
    ddr4_sdram_c0_act_n : out STD_LOGIC;
    ddr4_sdram_c0_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c0_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c0_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c0_ck_c : out STD_LOGIC;
    ddr4_sdram_c0_ck_t : out STD_LOGIC;
    ddr4_sdram_c0_cke : out STD_LOGIC;
    ddr4_sdram_c0_cs_n : out STD_LOGIC;
    ddr4_sdram_c0_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c0_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c0_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c0_odt : out STD_LOGIC;
    ddr4_sdram_c0_par : out STD_LOGIC;
    ddr4_sdram_c0_reset_n : out STD_LOGIC;
    default_300mhz_clk0_clk_n : in STD_LOGIC;
    default_300mhz_clk0_clk_p : in STD_LOGIC;
    ddr4_sdram_c1_act_n : out STD_LOGIC;
    ddr4_sdram_c1_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c1_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c1_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c1_ck_c : out STD_LOGIC;
    ddr4_sdram_c1_ck_t : out STD_LOGIC;
    ddr4_sdram_c1_cke : out STD_LOGIC;
    ddr4_sdram_c1_cs_n : out STD_LOGIC;
    ddr4_sdram_c1_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c1_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c1_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c1_odt : out STD_LOGIC;
    ddr4_sdram_c1_par : out STD_LOGIC;
    ddr4_sdram_c1_reset_n : out STD_LOGIC;
    default_300mhz_clk1_clk_n : in STD_LOGIC;
    default_300mhz_clk1_clk_p : in STD_LOGIC;
    ddr4_sdram_c2_act_n : out STD_LOGIC;
    ddr4_sdram_c2_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c2_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c2_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c2_ck_c : out STD_LOGIC;
    ddr4_sdram_c2_ck_t : out STD_LOGIC;
    ddr4_sdram_c2_cke : out STD_LOGIC;
    ddr4_sdram_c2_cs_n : out STD_LOGIC;
    ddr4_sdram_c2_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c2_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c2_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c2_odt : out STD_LOGIC;
    ddr4_sdram_c2_par : out STD_LOGIC;
    ddr4_sdram_c2_reset_n : out STD_LOGIC;
    default_300mhz_clk2_clk_n : in STD_LOGIC;
    default_300mhz_clk2_clk_p : in STD_LOGIC;
    ddr4_sdram_c3_act_n : out STD_LOGIC;
    ddr4_sdram_c3_adr : out STD_LOGIC_VECTOR ( 16 downto 0 );
    ddr4_sdram_c3_ba : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c3_bg : out STD_LOGIC_VECTOR ( 1 downto 0 );
    ddr4_sdram_c3_ck_c : out STD_LOGIC;
    ddr4_sdram_c3_ck_t : out STD_LOGIC;
    ddr4_sdram_c3_cke : out STD_LOGIC;
    ddr4_sdram_c3_cs_n : out STD_LOGIC;
    ddr4_sdram_c3_dq : inout STD_LOGIC_VECTOR ( 71 downto 0 );
    ddr4_sdram_c3_dqs_c : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c3_dqs_t : inout STD_LOGIC_VECTOR ( 17 downto 0 );
    ddr4_sdram_c3_odt : out STD_LOGIC;
    ddr4_sdram_c3_par : out STD_LOGIC;
    ddr4_sdram_c3_reset_n : out STD_LOGIC;
    default_300mhz_clk3_clk_n : in STD_LOGIC;
    default_300mhz_clk3_clk_p : in STD_LOGIC;
    pcie_perstn : in STD_LOGIC;
    resetn : in STD_LOGIC
  );
  end component top;
begin
top_i: component top
     port map (
      ddr4_sdram_c0_act_n => ddr4_sdram_c0_act_n,
      ddr4_sdram_c0_adr(16 downto 0) => ddr4_sdram_c0_adr(16 downto 0),
      ddr4_sdram_c0_ba(1 downto 0) => ddr4_sdram_c0_ba(1 downto 0),
      ddr4_sdram_c0_bg(1 downto 0) => ddr4_sdram_c0_bg(1 downto 0),
      ddr4_sdram_c0_ck_c => ddr4_sdram_c0_ck_c,
      ddr4_sdram_c0_ck_t => ddr4_sdram_c0_ck_t,
      ddr4_sdram_c0_cke => ddr4_sdram_c0_cke,
      ddr4_sdram_c0_cs_n => ddr4_sdram_c0_cs_n,
      ddr4_sdram_c0_dq(71 downto 0) => ddr4_sdram_c0_dq(71 downto 0),
      ddr4_sdram_c0_dqs_c(17 downto 0) => ddr4_sdram_c0_dqs_c(17 downto 0),
      ddr4_sdram_c0_dqs_t(17 downto 0) => ddr4_sdram_c0_dqs_t(17 downto 0),
      ddr4_sdram_c0_odt => ddr4_sdram_c0_odt,
      ddr4_sdram_c0_par => ddr4_sdram_c0_par,
      ddr4_sdram_c0_reset_n => ddr4_sdram_c0_reset_n,
      ddr4_sdram_c1_act_n => ddr4_sdram_c1_act_n,
      ddr4_sdram_c1_adr(16 downto 0) => ddr4_sdram_c1_adr(16 downto 0),
      ddr4_sdram_c1_ba(1 downto 0) => ddr4_sdram_c1_ba(1 downto 0),
      ddr4_sdram_c1_bg(1 downto 0) => ddr4_sdram_c1_bg(1 downto 0),
      ddr4_sdram_c1_ck_c => ddr4_sdram_c1_ck_c,
      ddr4_sdram_c1_ck_t => ddr4_sdram_c1_ck_t,
      ddr4_sdram_c1_cke => ddr4_sdram_c1_cke,
      ddr4_sdram_c1_cs_n => ddr4_sdram_c1_cs_n,
      ddr4_sdram_c1_dq(71 downto 0) => ddr4_sdram_c1_dq(71 downto 0),
      ddr4_sdram_c1_dqs_c(17 downto 0) => ddr4_sdram_c1_dqs_c(17 downto 0),
      ddr4_sdram_c1_dqs_t(17 downto 0) => ddr4_sdram_c1_dqs_t(17 downto 0),
      ddr4_sdram_c1_odt => ddr4_sdram_c1_odt,
      ddr4_sdram_c1_par => ddr4_sdram_c1_par,
      ddr4_sdram_c1_reset_n => ddr4_sdram_c1_reset_n,
      ddr4_sdram_c2_act_n => ddr4_sdram_c2_act_n,
      ddr4_sdram_c2_adr(16 downto 0) => ddr4_sdram_c2_adr(16 downto 0),
      ddr4_sdram_c2_ba(1 downto 0) => ddr4_sdram_c2_ba(1 downto 0),
      ddr4_sdram_c2_bg(1 downto 0) => ddr4_sdram_c2_bg(1 downto 0),
      ddr4_sdram_c2_ck_c => ddr4_sdram_c2_ck_c,
      ddr4_sdram_c2_ck_t => ddr4_sdram_c2_ck_t,
      ddr4_sdram_c2_cke => ddr4_sdram_c2_cke,
      ddr4_sdram_c2_cs_n => ddr4_sdram_c2_cs_n,
      ddr4_sdram_c2_dq(71 downto 0) => ddr4_sdram_c2_dq(71 downto 0),
      ddr4_sdram_c2_dqs_c(17 downto 0) => ddr4_sdram_c2_dqs_c(17 downto 0),
      ddr4_sdram_c2_dqs_t(17 downto 0) => ddr4_sdram_c2_dqs_t(17 downto 0),
      ddr4_sdram_c2_odt => ddr4_sdram_c2_odt,
      ddr4_sdram_c2_par => ddr4_sdram_c2_par,
      ddr4_sdram_c2_reset_n => ddr4_sdram_c2_reset_n,
      ddr4_sdram_c3_act_n => ddr4_sdram_c3_act_n,
      ddr4_sdram_c3_adr(16 downto 0) => ddr4_sdram_c3_adr(16 downto 0),
      ddr4_sdram_c3_ba(1 downto 0) => ddr4_sdram_c3_ba(1 downto 0),
      ddr4_sdram_c3_bg(1 downto 0) => ddr4_sdram_c3_bg(1 downto 0),
      ddr4_sdram_c3_ck_c => ddr4_sdram_c3_ck_c,
      ddr4_sdram_c3_ck_t => ddr4_sdram_c3_ck_t,
      ddr4_sdram_c3_cke => ddr4_sdram_c3_cke,
      ddr4_sdram_c3_cs_n => ddr4_sdram_c3_cs_n,
      ddr4_sdram_c3_dq(71 downto 0) => ddr4_sdram_c3_dq(71 downto 0),
      ddr4_sdram_c3_dqs_c(17 downto 0) => ddr4_sdram_c3_dqs_c(17 downto 0),
      ddr4_sdram_c3_dqs_t(17 downto 0) => ddr4_sdram_c3_dqs_t(17 downto 0),
      ddr4_sdram_c3_odt => ddr4_sdram_c3_odt,
      ddr4_sdram_c3_par => ddr4_sdram_c3_par,
      ddr4_sdram_c3_reset_n => ddr4_sdram_c3_reset_n,
      default_300mhz_clk0_clk_n => default_300mhz_clk0_clk_n,
      default_300mhz_clk0_clk_p => default_300mhz_clk0_clk_p,
      default_300mhz_clk1_clk_n => default_300mhz_clk1_clk_n,
      default_300mhz_clk1_clk_p => default_300mhz_clk1_clk_p,
      default_300mhz_clk2_clk_n => default_300mhz_clk2_clk_n,
      default_300mhz_clk2_clk_p => default_300mhz_clk2_clk_p,
      default_300mhz_clk3_clk_n => default_300mhz_clk3_clk_n,
      default_300mhz_clk3_clk_p => default_300mhz_clk3_clk_p,
      pci_express_x1_rxn => pci_express_x1_rxn,
      pci_express_x1_rxp => pci_express_x1_rxp,
      pci_express_x1_txn => pci_express_x1_txn,
      pci_express_x1_txp => pci_express_x1_txp,
      pcie_perstn => pcie_perstn,
      pcie_refclk_clk_n => pcie_refclk_clk_n,
      pcie_refclk_clk_p => pcie_refclk_clk_p,
      resetn => resetn
    );
end STRUCTURE;

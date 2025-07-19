# ./build/X86/gem5.opt --debug-flags=ExecAll --debug-file=ExecAll.out configs/deprecated/example/se.py -c tests/test-progs/linear-projection/bin/x86/linux/linear-projection
# ./build/X86/gem5.opt --debug-flags=ExecAll --debug-file=ExecAll.out configs/example/se.py -c tests/test-progs/hello/bin/x86/linux/hello 
# ./build/X86/gem5.opt --debug-flags=Config --debug-file=Config.out configs/deprecated/example/se.py -c tests/test-progs/linear-projection/bin/x86/linux/linear-projection --mem-type=NVMainMemory --caches --l2cache --l1i_size 32kB --l1d_size 32kB --l2_size 2MB --cpu-type=TimingSimpleCPU --nvmain-config=../RT-SIM/RTSimExport/Config/RM.config

# scons EXTRAS=../RT_SIM_IMC build/X86/gem5.opt -j32

./build/X86/gem5.opt \
--outdir=x86/m5out_SVHN_32_trans_pim \
--debug-flags=NVMainMyDebug,TLB --debug-file=NVMainMyDebug.out \
configs/deprecated/example/se_x86.py \
-c tests/test-progs/transformer_pim/bin/x86/linux/transformer_pim \
--mem-type=NVMainMemory \
--caches --l2cache --l1i_size 32kB --l1d_size 32kB --l2_size 2MB \
--cpu-type=X86MinorCPU \
--cpu-clock=3.2GHz \
--sys-clock=400MHz \
--nvmain-config=../RT_SIM_IMC/Config/SK.config

# X86O3CPU
# X86TimingSimpleCPU



# --cpu-type=MinorCPU \
# --cpu-type=AtomicSimpleCPU \
# --caches --l2cache --l1i_size 32kB --l1d_size 32kB --l2_size 512kB \
# --caches --l1i_size 16kB --l1d_size 16kB \

# --kernel=../.cache/gem5/x86-linux-kernel-5.4.49  \
# --disk-image=../.cache/gem5/x86-ubuntu-18.04-img  \
# --script=tests/test-progs/transformer_pim/src/script.sh \

# -c tests/test-progs/transformer_pim/bin/x86/linux/transformer_pim \
# --debug-flags=ExecEnable,ExecResult,ExecKernel,ExecEffAddr,ExecOpClass,ExecUser,ExecMacro,ExecFaulting --debug-file=Exec.out \
# Exec = ExecEnable,ExecOpClass,ExecThread,ExecEffAddr,ExecResult,ExecSymbol,ExecMicro,ExecMacro,ExecFaulting,ExecUser,ExecKernel
# transformer_sim
# mmap_test_v2
# IMC_test
# --debug-flags=CacheAll --debug-file=CacheAll.out \

# NVMain
# NVMainMin
# NVMainMyDebug

# CfiMemory,DDR3_1600_8x8,DDR3_2133_8x8,DDR4_2400_16x4,DDR4_2400_4x16,DDR4_2400_8x8,DDR5_4400_4x8,DDR5_6400_4x8,
# DDR5_8400_4x8,DRAMInterface,GDDR5_4000_2x32,HBM_1000_4H_1x128,HBM_1000_4H_1x64,HBM_2000_4H_1x64,HMC_2500_1x32,
# LPDDR2_S4_1066_1x32,LPDDR3_1600_1x32,LPDDR5_5500_1x16_8B_BL32,LPDDR5_5500_1x16_BG_BL16,LPDDR5_5500_1x16_BG_BL32,
# LPDDR5_6400_1x16_8B_BL32,LPDDR5_6400_1x16_BG_BL16,LPDDR5_6400_1x16_BG_BL32,NVMInterface,NVM_2400_1x64,NVMainMemory,
# QoSMemSinkInterface,SimpleMemory,WideIO_200_1x128

# [--cpu-type {AtomicSimpleCPU,BaseAtomicSimpleCPU,BaseMinorCPU,BaseNonCachingSimpleCPU,BaseO3CPU,BaseTimingSimpleCPU,
# DerivO3CPU,NonCachingSimpleCPU,O3CPU,TimingSimpleCPU,X86AtomicSimpleCPU,X86KvmCPU,X86MinorCPU,X86NonCachingSimpleCPU,
# X86O3CPU,X86TimingSimpleCPU}]

# Base Flags:
#     AMBA:
#     Activity:
#     AddrRanges:
#     Annotate: State machine annotation debugging
#     AnnotateQ: State machine annotation queue debugging
#     AnnotateVerbose: Dump all state machine annotation details
#     Arm:
#     ArmTme: Transactional Memory Extension
#     BTB:
#     BaseXBar:
#     Branch:
#     Bridge:
#     CCRegs:
#     CFI:
#     Cache:
#     CacheComp:
#     CachePort:
#     CacheRepl:
#     CacheTags:
#     CacheVerbose:
#     Checker:
#     Checkpoint:
#     ClockDomain:
#     CoherentXBar:
#     CommMonitor:
#     Commit:
#     CommitRate:
#     Config:
#     Context:
#     CxxConfig:
#     DMA:
#     DMACopyEngine:
#     DRAM:
#     DRAMPower:
#     DRAMSim2:
#     DRAMState:
#     DRAMsim3:
#     DVFS:
#     DebugPrintf:
#     Decode:
#     Decoder: Decoder debug output
#     DirectedTest:
#     DiskImageRead:
#     DiskImageWrite:
#     DistEthernet:
#     DistEthernetCmd:
#     DistEthernetPkt:
#     Drain:
#     DynInst:
#     EnergyCtrl:
#     Ethernet:
#     EthernetCksum:
#     EthernetDMA:
#     EthernetData:
#     EthernetDesc:
#     EthernetEEPROM:
#     EthernetIntr:
#     EthernetPIO:
#     EthernetSM:
#     Event:
#     ExecAsid: Format: Include ASID in trace
#     ExecCPSeq: Format: Instruction sequence number
#     ExecEffAddr: Format: Include effective address
#     ExecEnable: Filter: Enable exec tracing (no tracing without this)
#     ExecFaulting: Trace faulting instructions
#     ExecFetchSeq: Format: Fetch sequence number
#     ExecFlags: Format: Include instruction flags in trace
#     ExecKernel: Filter: Trace kernel mode instructions
#     ExecMacro: Filter: Include macroops
#     ExecMicro: Filter: Include microops
#     ExecOpClass: Format: Include operand class
#     ExecRegDelta:
#     ExecResult: Format: Include results from execution
#     ExecSymbol: Format: Try to include symbol names
#     ExecThread: Format: Include thread ID in trace
#     ExecUser: Filter: Trace user mode instructions
#     ExternalPort:
#     FVPBasePwrCtrl:
#     Faults: Information about faults, exceptions, interrupts, etc
#     Fetch:
#     FlashDevice:
#     FloatRegs:
#     Flow:
#     FreeList:
#     GDBAcc: Remote debugger accesses
#     GDBExtra: Dump extra information on reads and writes
#     GDBMisc: Breakpoints, traps, watchpoints, etc.
#     GDBRead: Reads to the remote address space
#     GDBRecv: Messages received from the remote application
#     GDBSend: Messages sent to the remote application
#     GDBWrite: Writes to the remote address space
#     GIC:
#     GICV2M:
#     GUPSGen:
#     GarnetSyntheticTraffic:
#     HDLcd:
#     HMCController:
#     HWPrefetch:
#     HWPrefetchQueue:
#     HelloExample: For Learning gem5 Part 2. Simple example debug flag
#     HtmCpu: Hardware Transactional Memory (CPU side)
#     HtmMem: Hardware Transactional Memory (Mem side)
#     IEW:
#     IPI:
#     IPR:
#     IQ:
#     ITS:
#     IdeCtrl:
#     IdeDisk:
#     Indirect:
#     IntRegs:
#     Intel8254Timer:
#     Interrupt:
#     InvalidReg:
#     IsaFake:
#     Kvm: Basic KVM Functionality
#     KvmContext: KVM/gem5 context synchronization
#     KvmIO: KVM MMIO diagnostics
#     KvmInt: KVM Interrupt handling
#     KvmRun: KvmRun entry/exit diagnostics
#     KvmTimer: KVM timing
#     LLSC:
#     LSQ:
#     LSQUnit:
#     LTage:
#     Loader:
#     LupioBLK:
#     LupioIPI:
#     LupioPIC:
#     LupioRNG:
#     LupioRTC:
#     LupioSYS:
#     LupioTMR:
#     LupioTTY:
#     MC146818:
#     MHU:
#     MMU:
#     MPAM: MPAM debug flag
#     MSHR:
#     MatRegs:
#     MemChecker:
#     MemCheckerMonitor:
#     MemCtrl:
#     MemDepUnit:
#     MemTest:
#     MemoryAccess:
#     MinorCPU: Minor CPU-level events
#     MinorExecute: Minor Execute stage
#     MinorInterrupt: Minor interrupt handling
#     MinorMem: Minor memory accesses
#     MinorScoreboard: Minor Execute register scoreboard
#     MinorTiming: Extra timing for instructions
#     MinorTrace: MinorTrace cycle-by-cycle state trace
#     MiscRegs:
#     Mwait:
#     NVM:
#     NVMain:
#     NVMainMin:
#     NVMainMyDebug:
#     NoMali:
#     NoncoherentXBar:
#     O3CPU:
#     O3PipeView:
#     PCEvent:
#     PL111:
#     PMUVerbose: Performance Monitor
#     PS2:
#     PacketQueue:
#     PageTableWalker: Page table walker state machine debugging
#     PartitionPolicy:
#     PcCountTracker:
#     PciDevice:
#     PciHost:
#     Pl050:
#     PortTrace:
#     PowerDomain:
#     Printf:
#     ProbeVerbose:
#     ProtocolTrace:
#     PseudoInst:
#     QOS:
#     QemuFwCfg:
#     QemuFwCfgVerbose:
#     Quiesce:
#     RAS:
#     ROB:
#     RVCTRL:
#     Rename:
#     ResponsePort:
#     RubyCache:
#     RubyCacheTrace:
#     RubyDma:
#     RubyGenerated:
#     RubyHitMiss:
#     RubyNetwork:
#     RubyPort:
#     RubyPrefetcher:
#     RubyProtocol:
#     RubyQueue:
#     RubyResourceStalls:
#     RubySequencer:
#     RubySlicc:
#     RubyStats:
#     RubySystem:
#     RubyTest:
#     RubyTester:
#     SCMI:
#     SMMUv3:
#     SMMUv3Hazard:
#     SQL: SQL queries sent to the server
#     Scoreboard:
#     Semihosting:
#     SerialLink:
#     SimpleCPU:
#     SimpleCache: For Learning gem5 Part 2.
#     SimpleDisk:
#     SimpleDiskData:
#     SimpleMemobj: For Learning gem5 Part 2.
#     SimpleTrace:
#     SnoopFilter:
#     Sp805:
#     SpatterGen:
#     SpatterKernel:
#     Stack:
#     StackDist:
#     StatEvents: Statistics event tracking
#     Stats: Statistics management
#     StoreSet:
#     SysBridge:
#     SyscallBase:
#     SyscallVerbose:
#     TLB:
#     TLBVerbose:
#     Tage:
#     TageSCL:
#     Terminal:
#     TerminalVerbose:
#     ThermalDomain:
#     Thread:
#     TimeSync:
#     Timer:
#     TlmBridge:
#     TokenPort:
#     TraceCPUData:
#     TraceCPUInst:
#     TrafficGen:
#     UFSHostDevice:
#     Uart:
#     VGIC:
#     VIO: VirtIO base functionality
#     VIO9P: General 9p over VirtIO debugging
#     VIO9PData: Dump data in VirtIO 9p connections
#     VIOBlock: VirtIO block device
#     VIOConsole: VirtIO console device
#     VIOIface: VirtIO transport
#     VIORng: VirtIO entropy source device
#     VNC:
#     VecPredRegs:
#     VecRegs:
#     Vma:
#     VoltageDomain:
#     VtoPhys:
#     WorkItems:
#     Writeback:

# Compound Flags:
#     AnnotateAll: All Annotation flags
#         Annotate, AnnotateQ, AnnotateVerbose
#     CacheAll:
#         Cache, CacheComp, CachePort, CacheRepl, CacheVerbose, HWPrefetch,
#         MSHR, PartitionPolicy
#     DiskImageAll:
#         DiskImageRead, DiskImageWrite
#     EthernetAll:
#         Ethernet, EthernetPIO, EthernetDMA, EthernetData, EthernetDesc,
#         EthernetIntr, EthernetSM, EthernetCksum, EthernetEEPROM
#     EthernetNoData:
#         Ethernet, EthernetPIO, EthernetDesc, EthernetIntr, EthernetSM,
#         EthernetCksum
#     Exec:
#         ExecEnable, ExecOpClass, ExecThread, ExecEffAddr, ExecResult,
#         ExecSymbol, ExecMicro, ExecMacro, ExecFaulting, ExecUser, ExecKernel
#     ExecAll:
#         ExecEnable, ExecCPSeq, ExecEffAddr, ExecFaulting, ExecFetchSeq,
#         ExecOpClass, ExecRegDelta, ExecResult, ExecSymbol, ExecThread,
#         ExecMicro, ExecMacro, ExecUser, ExecKernel, ExecAsid, ExecFlags
#     ExecNoTicks:
#         Exec, FmtTicksOff
#     GDBAll: All Remote debugging flags
#         GDBMisc, GDBAcc, GDBRead, GDBWrite, GDBSend, GDBRecv, GDBExtra
#     IdeAll:
#         IdeCtrl, IdeDisk
#     KvmAll: All KVM debug flags
#         Kvm, KvmContext, KvmRun, KvmIO, KvmInt, KvmTimer
#     Minor:
#         MinorCPU, MinorExecute, MinorInterrupt, MinorMem, MinorScoreboard
#     O3CPUAll:
#         Fetch, Decode, Rename, IEW, Commit, IQ, ROB, FreeList, LSQ, LSQUnit,
#         StoreSet, MemDepUnit, DynInst, O3CPU, Activity, Scoreboard,
#         Writeback
#     Registers:
#         IntRegs, FloatRegs, VecRegs, VecPredRegs, MatRegs, CCRegs, MiscRegs
#     Ruby:
#         RubyQueue, RubyNetwork, RubyTester, RubyGenerated, RubySlicc,
#         RubySystem, RubyCache, RubyDma, RubyPort, RubySequencer,
#         RubyCacheTrace, RubyPrefetcher, RubyProtocol, RubyHitMiss
#     SyscallAll:
#         SyscallBase, SyscallVerbose
#     XBar:
#         BaseXBar, CoherentXBar, NoncoherentXBar, SnoopFilter

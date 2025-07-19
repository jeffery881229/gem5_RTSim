#########################
# x86-ubuntu-run.py
#########################
from gem5.utils.requires import requires
from gem5.components.boards.x86_board import X86Board
from gem5.components.memory.single_channel import SingleChannelDDR3_1600
from gem5.components.cachehierarchies.ruby.mesi_two_level_cache_hierarchy import (
    MESITwoLevelCacheHierarchy,
)
from gem5.components.processors.simple_switchable_processor import (
    SimpleSwitchableProcessor,
)
from gem5.coherence_protocol import CoherenceProtocol
from gem5.isas import ISA
from gem5.components.processors.cpu_types import CPUTypes
from gem5.resources.resource import Resource
from gem5.simulate.simulator import Simulator
from gem5.simulate.exit_event import ExitEvent

#
# 1. 要求 binary 要支援 X86 + MESI_TWO_LEVEL + KVM
#    若無KVM可把 kvm_required=True 移除
#
requires(
    isa_required=ISA.X86,
    coherence_protocol_required=CoherenceProtocol.MESI_TWO_LEVEL,
    # kvm_required=True,  
)

# 2. Cache: MESI two-level (ruby)
cache_hierarchy = MESITwoLevelCacheHierarchy(
    l1d_size="32KiB", l1d_assoc=8,
    l1i_size="32KiB", l1i_assoc=8,
    l2_size="256kB", l2_assoc=16,
    num_l2_banks=1
)

# 3. 記憶體：單通道 DDR3_1600, 注意大小不要超過 3GiB
memory = SingleChannelDDR3_1600(size="2GiB")

# 4. 處理器：用 SimpleSwitchableProcessor, 先KVM,再TIMING, 2 cores
processor = SimpleSwitchableProcessor(
    starting_core_type=CPUTypes.TIMING,
    switch_core_type=CPUTypes.TIMING,
    num_cores=2,
)

# 5. 建立 X86 board, full-system
board = X86Board(
    clk_freq="3GHz",
    processor=processor,
    memory=memory,
    cache_hierarchy=cache_hierarchy
)

#
# 6. readfile_contents: 要在啟動 Ubuntu 後執行的指令
#    先 `m5 exit` 用於暫停 => 由 on_exit_event 觸發 processor.switch() => TIMING
#    再 echo & sleep => 再次 m5 exit => 結束
#
command = "m5 exit;" \
          "echo 'This is running on Timing CPU cores.';" \
          "sleep 1;" \
          "m5 exit;"

#
# 7. 設定 kernel+disk (官方 resource)
#    預設會自動下載 kernel= x86-linux-kernel-5.4.49 & x86-ubuntu-18.04-img
#
board.set_kernel_disk_workload(
    kernel=Resource("x86-linux-kernel-5.4.49"),
    disk_image=Resource("x86-ubuntu-18.04-img"),
    readfile_contents=command
)

#
# 8. 建立 Simulator, 改寫 on_exit_event => 第一次 EXIT 時改成切換core
#
simulator = Simulator(
    board=board,
    on_exit_event={
        ExitEvent.EXIT : (func() for func in [processor.switch]),
    },
)

# 9. 執行
simulator.run()

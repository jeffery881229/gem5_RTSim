import sys
sys.path.insert(0, "/RAID2/LAB/css/cssRA01/gem5_arm/build/ARM/python")
sys.path.insert(0, "/RAID2/LAB/css/cssRA01/gem5_arm/configs")

from m5.objects import *
from common.cores.arm.O3_ARM_v7a import O3_ARM_v7a_3

print("Success:", O3_ARM_v7a_3)

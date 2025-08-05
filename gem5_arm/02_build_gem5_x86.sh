# sed -i '36s/^#//' /home/csslab438/RT-SIM/RTSimExport/SConscript
# python3 `which scons` EXTRAS=../RTSimExport build/X86/gem5.opt -j32
scons EXTRAS=../RT_SIM_IMC build/X86/gem5.opt -j8

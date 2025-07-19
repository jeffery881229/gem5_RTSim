# g++ -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11
# arm-linux-gnueabihf-g++ -o ../bin/x86/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11
arm-linux-gnueabihf-g++ -static -o ../bin/arm/linux/transformer_pim transformer_pim.cpp -pthread -std=c++11

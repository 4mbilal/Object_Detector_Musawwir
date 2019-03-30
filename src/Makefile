CC=g++
CFLAGS=-O2 -g `pkg-config --cflags opencv`	-mcpu=cortex-a7 -mfpu=neon-vfpv4 -mfloat-abi=hard
LDFLAGS=`pkg-config --libs opencv`

BIN=Obj_Detector
Obj_Detector : Defines.cpp NMS.cpp HSG.cpp Object_Detector.cpp Main.cpp	TwoD_Filters.cpp	Image_Utilities.cpp

	$(CC) $(CFLAGS) $(LDFLAGS) -o $(BIN) Defines.cpp NMS.cpp HSG.cpp Object_Detector.cpp Main.cpp	TwoD_Filters.cpp	Image_Utilities.cpp
	./Obj_Detector


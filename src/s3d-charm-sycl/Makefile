#===============================================================================
# User Options
#===============================================================================

# Compiler can be set below, or via environment variable
CC        = cscc
OPTIMIZE  = yes
DEBUG     = no
LAUNCHER  =

#===============================================================================
# Program name & source code list
#===============================================================================

program = main

source = main.cpp Option.cpp OptionParser.cpp S3D_with_nd_range.cpp

obj = $(source:.cpp=.o)

#===============================================================================
# Targets to Build
#===============================================================================

$(program): $(obj)
        $(CC) $(CFLAGS) $(obj) -o $@ $(LDFLAGS)

%.o: %.cpp
        $(CC) $(CFLAGS) -c $< -o $@

main.o: main.cpp OptionParser.h Utility.h

Option.o: Option.cpp Option.h

OptionParser.o: OptionParser.cpp OptionParser.h Utility.h

S3D_with_nd_range.o: S3D_with_nd_range.cpp S3D.h OptionParser.h *.sycl

clean:
        rm -rf $(program) $(obj)

run: $(program)
        $(LAUNCHER) ./$(program) -q -n 100 -s 1
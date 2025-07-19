import subprocess
import os
import sys
import pathlib

config = 'Config/SK-external-2048.config'
max_cycles = "10000000000000000000000000000000000000000000000000000000000000000000000000"

def get_nvts(directory: str) -> list:
    files = [f for f in os.listdir(directory) if f.endswith(".nvt")]
    return files

def run_file(file: str, output_file: str) -> bool:
    print("Processing ", file, " to ", output_file)
    print("####################### Running RTSIM #######################")
    if "skye" in file:
        config='Config/SK-external-2048.config'
    else:
        config='Config/SK-external-4096.config'
    print(config)
    result = subprocess.run(['./nvmain.fast', config, file, max_cycles], stdout=subprocess.PIPE)
    result = result.stdout.decode('utf-8').splitlines()
    output_data = [line for line in result if "i0.defaultMemory.channel0.RTM" in line]
    print("####################### RTSIM  DONE, WRITING OUTPUT #######################")
    with open(output_file, "w") as f:        
        f.writelines(line + '\n' for line in output_data)

    print("####################### WRITING OUTPUT DONE #######################")
    return True


def main() -> int:
    args = sys.argv
    
    #Check if directory with nvts
    if len(args) < 2:
        print("Missing source")
        return -1

    sources = []
    source_directory = pathlib.Path(__file__).parent / args[1] 

    try:
        directory = pathlib.Path(__file__).parent / args[1]        
        sources = get_nvts(directory)        
    except FileNotFoundError:
        print("Source directory not found")
        return -1

    if len(sources) < 1:
        print("No .NVTs found")

    #Check if result destination present
    if len(args) < 3:
        print("Missing destination")
        return -1

    results_directory = pathlib.Path(__file__).parent / args[2]         

    for source in sources:
        source_file = pathlib.Path(__file__).parent / args[1] / source
        result_file = pathlib.Path(__file__).parent / args[2] / source.replace(".nvt",".txt")
        print(result_file)
        run_file(str(source_file), str(result_file))

    return 0

if __name__ == '__main__':
    sys.exit(main())
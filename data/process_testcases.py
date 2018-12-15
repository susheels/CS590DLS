# create dataset labels (Vulnerable  (bad) not Not-Vulnerable (good))
# input data as folders of testcases

import os
import sys
import subprocess
import glob



def _createdir(dname):
    try:
        os.makedirs(dname)
    except FileExistsError:
        pass


def convert_2_llvm_good(file_path,output_llvm_dir_path_good,program_type):

	f, file_extension = os.path.splitext(file_path)
	filename = f.split("/")[-1]

	output_file_path = output_llvm_dir_path_good + filename +"_"+program_type+"_"+".ll"
	compiler_inst = "clang -S -w -emit-llvm -D OMITBAD "+file_path+" -o "+output_file_path
	subprocess.call([compiler_inst],shell=True)
	
def convert_2_llvm_bad(file_path,output_llvm_dir_path_bad,program_type):

	f, file_extension = os.path.splitext(file_path)
	filename = f.split("/")[-1]

	output_file_path = output_llvm_dir_path_bad + filename +"_"+program_type+"_"+".ll"
	compiler_inst = "clang -S -w -emit-llvm -D OMITGOOD "+file_path+" -o "+output_file_path
	subprocess.call([compiler_inst],shell=True)

if __name__ == '__main__':	

	
	file_dir_path = "C/testcases/"
	output_llvm_dir_path_good = "llvm_ir/1/"
	output_llvm_dir_path_bad = "llvm_ir/0/"

	_createdir(output_llvm_dir_path_good)
	_createdir(output_llvm_dir_path_bad)

	list_of_c_file_paths = glob.glob(file_dir_path+"/**/*.c", recursive=True)
	list_of_cpp_file_paths = glob.glob(file_dir_path+"/**/*.cpp", recursive=True)
	for file in list_of_cpp_file_paths:
		os.remove(file)
	

	
	print("Number of C Source Code Files: ",len(list_of_c_file_paths))
	c = 1
	for filename_path in list_of_c_file_paths:
		if c == 2000:
			break
		c += 1
		omit_w32 = False
		if(filename_path.find("w32") == -1):
			omit_w32 = True

		if(omit_w32 and (filename_path.find("good") != -1)):
			convert_2_llvm_good(filename_path,output_llvm_dir_path_good,"c")
			continue
		elif(omit_w32 and (filename_path.find("bad") != -1)):
			convert_2_llvm_bad(filename_path,output_llvm_dir_path_bad,"c")
			continue
		elif(omit_w32):
			convert_2_llvm_good(filename_path,output_llvm_dir_path_good,"c")
			convert_2_llvm_bad(filename_path,output_llvm_dir_path_bad,"c")

		if c % 500 == 0:
			print("LLVM IR generated for C:",c, "of", len(list_of_c_file_paths))
	
	del list_of_c_file_paths

	print("Number of C++ Source Code Files: ",len(list_of_cpp_file_paths))
	c = 1
	for filename_path in list_of_cpp_file_paths:
		c += 1
		omit_w32 = False
		if(filename_path.find("w32") == -1):
			omit_w32 = True

		if(omit_w32 and (filename_path.find("good") != -1)):
			convert_2_llvm_good(filename_path,output_llvm_dir_path_good,"cpp")
			continue
		elif(omit_w32 and (filename_path.find("bad") != -1)):
			convert_2_llvm_bad(filename_path,output_llvm_dir_path_bad,"cpp")
			continue
		elif(omit_w32):
			convert_2_llvm_good(filename_path,output_llvm_dir_path_good,"cpp")
			convert_2_llvm_bad(filename_path,output_llvm_dir_path_bad,"cpp")
		if c % 500 == 0:
			print("LLVM IR generated for C++:",c, "of", len(list_of_cpp_file_paths))


	
	
	
	
	


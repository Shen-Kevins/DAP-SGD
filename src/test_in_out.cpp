#include<iostream>
#include<fstream>

using namespace std;

int main(){
    fstream outfile("test.txt",ios::out);
    
    outfile<<"hello world\n";

    outfile.close();
    return 0;
}
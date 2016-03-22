#include <string>
#include <sstream>
#include <vector>
#include <iostream>

int main(){
    string a = "1.2@2.3@4.4@0.3";
    char * pch;
    pch = strtok(a.c_str(), "@");
    while ( pch != NULL){
        std::cout<<stof(pch));
	pch = strtok(NULL, "@");
    }
    return  0;
}

#include <bits/stdc++.h>
using namespace std;

int main(){
    for(int i=0 ; i < 28 ; i++){
        for(int j=0 ; j < 28 ; j++){
            int x = (i+j)*255/54;
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
            cout<<"\033[48;2;"<<x<<";"<<x<<";"<<x<<"m ";
        }
        cout << "\033[0m\n";
    }
}
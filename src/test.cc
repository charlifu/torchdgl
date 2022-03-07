#include <iostream>

#include <ATen/core/TensorBody.h>
#include <torch/custom_class.h>
#include <torch/script.h>

using namespace std;

vector<int64_t> foo() {
    vector<int64_t> vec;
    vec.push_back(1);
    vec.push_back(2);
    c10::IntArrayRef s(vec);
    cout << s << endl;
    return s.vec();
}

int main(int argc, char **argv)
{
    c10::IntArrayRef rst = foo();
    cout << rst << endl;
    return 0;
}
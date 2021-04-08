
#include <serialize/graph_serialize.h>
#include <torchdglgraph/torch_dgl_graph.h>

#include <ATen/Functions.h>
#include <iostream>
#include <memory>
#include <torch/script.h> // One-stop header.

int main(int argc, const char *argv[]) {
  if (argc != 2) {
    std::cerr << "usage: example-app <path-to-exported-script-module>\n";
    return -1;
  }

  torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    module = torch::jit::load(argv[1]);
  } catch (const c10::Error &e) {
    std::cerr << "error loading the model\n";
    return -1;
  }
  auto glist = dgl::serialize::LoadHeteroGraphs("./graph.bin", {});
  auto hgptr = glist[0]->gptr;
  std::vector<at::IValue> input_list;
  TorchDGLGraph tg(hgptr);
  input_list.push_back(torch::make_custom_class<TorchDGLGraph>(hgptr));
  input_list.push_back(torch::arange(5, 10));
  auto output = module.forward(input_list);
  std::cout << output.toTensor() << std::endl;

  std::cout << "ok\n";
}
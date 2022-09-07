#include <torch/script.h>
#include <torch/torch.h>
#include<python.h>

static struct PyModuleDef _op = {
    PyModuleDef_HEAD_INIT,
    "_op",
    "_op doc",
    -1,
    NULL,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_op(void)
{
    return PyModule_Create(&_op);
}

torch:: Tensor repeatedInterleave(torch:: Tensor input){
   const int64_t N = input.sizes()[0];
   float* input_data = input.data_ptr<float>();
   const int64_t repeat = 512/N;
   torch:: Tensor output = torch::zeros(512);
   float* out = output.data_ptr<float>();
   auto k = 0;
   for (auto i = 0; i < N ; i++) {
    for (auto j = 0 ; j < repeat; j++){
      out[k] = input_data[i].item();
      k = k + 1;
    }

  }
  return output;
}

torch::Tensor op(torch::Tensor layerOne, torch::Tensor layerTwo, torch::Tensor layerThree, torch::Tensor layerFour) {
  
  torch::Tensor layerOne_512 = torch::zeros({512});
  torch::Tensor layerTwo_512 = torch::zeros({512});
  torch::Tensor layerThree_512 = torch::zeros({512});
  torch::Tensor layerFour_512 = torch::zeros({512});
  torch::Tensor embedding = torch::zeros({512});
  
  float* layerOne_pointer = layerOne_512.data_ptr<float>();
  float* layerTwo_pointer = layerTwo_512.data_ptr<float>();
  float* layerThree_pointer = layerThree_512.data_ptr<float>();
  float* layerFour_pointer = layerFour_512.data_ptr<float>();
  float* embedding_pointer = embedding.data_ptr<float>();
  
  layerOne_512 = repeatedInterleave(layerOne);
  layerTwo_512 = repeatedInterleave(layerTwo);
  layerThree_512 =repeatedInterleave(layerThree);
  layerFour_512 =repeatedInterleave(layerFour);

  for(auto i=0; i< 512; i++){
    embedding_pointer[i] = (layerOne_pointer[i].item()+layerTwo_pointer[i].item()+layerThree_pointer[i].item()+layerFour_pointer[i].item())/4;
  }

  return embedding.clone();

}


static auto registry = torch::RegisterOperators("custom_namespace::op", &op);


#include <fstream>
#include <vector>
#include <memory>

extern std::vector<unsigned char> train_labels, test_labels;
extern std::vector<std::vector<float>> train_images, test_images;




struct init_cifar{

    init_cifar()
    {
        char file_name[6][20] = {"data_batch_1.bin","data_batch_2.bin","data_batch_3.bin",
                               "data_batch_4.bin","data_batch_5.bin","test_batch.bin  "};
        for(int i = 0;i < 6;++i)
        {
            std::ifstream in(file_name[i],std::ios::in | std::ios::binary);
            if(!in)
                throw std::runtime_error("Cannot open CIFAR");
            for(int j = 0;j < 10000;++j)
            {
                std::vector<unsigned char> buf(3073);
                std::vector<float> I(3072);
                in.read((char*)&buf[0],3073);

                for(int k = 0;k < I.size();++k)
                    I[k] = ((float)buf[k+1]/255.0f)*2.0f-1.0f;
                if(i == 5)
                {
                    test_labels.push_back(buf[0]);
                    test_images.push_back(I);
                }
                else
                {
                    train_labels.push_back(buf[0]);
                    train_images.push_back(I);
                }
            }
        }

    }
};

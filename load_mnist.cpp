#include <fstream>
#include <vector>
#include <memory>

std::vector<unsigned char> train_labels, test_labels;
std::vector<std::vector<float>> train_images, test_images;


void parse_mnist_labels(const std::string& label_file, std::vector<unsigned char> *labels);
void parse_mnist_images(const std::string& image_file,std::vector<std::vector<float>> *images,float scale_min,float scale_max,int x_padding,int y_padding);


struct init_mnist{

    init_mnist()
    {
        parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
        parse_mnist_images("train-images.idx3-ubyte", &train_images, -1.0, 1.0, 2, 2);
        parse_mnist_labels("t10k-labels.idx1-ubyte", &test_labels);
        parse_mnist_images("t10k-images.idx3-ubyte", &test_images, -1.0, 1.0, 2, 2);
    }
} ;//imp;


struct mnist_header {
    uint32_t magic_number;
    uint32_t num_items;
    uint32_t num_rows;
    uint32_t num_cols;
};

template<class T>
T* reverse_endian(T* p) {
    std::reverse(reinterpret_cast<char*>(p), reinterpret_cast<char*>(p) + sizeof(T));
    return p;
}


inline bool is_little_endian() {
    int x = 1;
    return *(char*) &x != 0;
}
void parse_mnist_header(std::ifstream& ifs, mnist_header& header) {
    ifs.read((char*) &header.magic_number, 4);
    ifs.read((char*) &header.num_items, 4);
    ifs.read((char*) &header.num_rows, 4);
    ifs.read((char*) &header.num_cols, 4);

    if (is_little_endian()) {
        reverse_endian(&header.magic_number);
        reverse_endian(&header.num_items);
        reverse_endian(&header.num_rows);
        reverse_endian(&header.num_cols);
    }

    if (header.magic_number != 0x00000803 || header.num_items <= 0)
        throw std::exception("MNIST label-file format error");
    if (ifs.fail() || ifs.bad())
        throw std::exception("file error");
}



void parse_mnist_labels(const std::string& label_file, std::vector<unsigned char> *labels)
{
    std::ifstream ifs(label_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw std::exception("failed to open file");

    uint32_t magic_number, num_items;

    ifs.read((char*) &magic_number, 4);
    ifs.read((char*) &num_items, 4);

    if (is_little_endian()) { // MNIST data is big-endian format
        reverse_endian(&magic_number);
        reverse_endian(&num_items);
    }

    if (magic_number != 0x00000801 || num_items <= 0)
        throw std::exception("MNIST label-file format error");

    for (uint32_t i = 0; i < num_items; i++) {
        uint8_t label;
        ifs.read((char*) &label, 1);
        labels->push_back((unsigned char) label);
    }
}

void parse_mnist_image(std::ifstream& ifs,
    const mnist_header& header,
    float scale_min,
    float scale_max,
    int x_padding,
    int y_padding,
    std::vector<float>& dst) {
    const int width = header.num_cols + 2 * x_padding;
    const int height = header.num_rows + 2 * y_padding;

    std::vector<uint8_t> image_vec(header.num_rows * header.num_cols);

    ifs.read((char*) &image_vec[0], header.num_rows * header.num_cols);

    dst.resize(width * height, scale_min);

    for (uint32_t y = 0; y < header.num_rows; y++)
      for (uint32_t x = 0; x < header.num_cols; x++)
        dst[width * (y + y_padding) + x + x_padding]
        = (image_vec[y * header.num_cols + x] / float(255)) * (scale_max - scale_min) + scale_min;
}

void parse_mnist_images(const std::string& image_file,
    std::vector<std::vector<float>> *images,
    float scale_min,
    float scale_max,
    int x_padding,
    int y_padding)
{

    if (x_padding < 0 || y_padding < 0)
        throw std::exception("padding size must not be negative");
    if (scale_min >= scale_max)
        throw std::exception("scale_max must be greater than scale_min");

    std::ifstream ifs(image_file.c_str(), std::ios::in | std::ios::binary);

    if (ifs.bad() || ifs.fail())
        throw std::exception("failed to open file:");

    mnist_header header;

    parse_mnist_header(ifs, header);

    for (uint32_t i = 0; i < header.num_items; i++) {
        std::vector<float> image;
        parse_mnist_image(ifs, header, scale_min, scale_max, x_padding, y_padding, image);
        images->push_back(image);
    }
}


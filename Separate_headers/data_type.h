typedef float field_t;
typedef float data_t;
typedef unsigned char uint8_t;
typedef unsigned int uint;

template<typename T>
struct data_arr
{
  T *d;
  size_t len;
};

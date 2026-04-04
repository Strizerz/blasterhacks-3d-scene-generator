/* Override MSVC version macros to satisfy CUDA 12.1 host_config.h check (_MSC_VER < 1940) */
#ifdef _MSC_VER
#undef _MSC_VER
#define _MSC_VER 1939
#endif
#ifdef _MSC_FULL_VER
#undef _MSC_FULL_VER
#define _MSC_FULL_VER 193900000
#endif

In file included from /Users/ersibesi/Desktop/LLAMA/llama.cpp/src/llama.cpp:16:
/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/src/../include/tensor_trace.h:53:1: warning: '_Static_assert' is a C11 extension [-Wc11-extensions]
_Static_assert(sizeof(struct TensorAccessLog) == 64,
^
In file included from /Users/ersibesi/Desktop/LLAMA/llama.cpp/src/llama-model.cpp:16:
/Users/ersibesi/Desktop/LLAMA/llama.cpp/ggml/src/../include/tensor_trace.h:53:1: warning: '_Static_assert' is a C11 extension [-Wc11-extensions]
_Static_assert(sizeof(struct TensorAccessLog) == 64,


/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:35:5: warning: no previous prototype for function 'extract_layer_id' [-Wmissing-prototypes]
int extract_layer_id(const char* name) {
    ^
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:35:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
int extract_layer_id(const char* name) {
^
static 
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:47:13: warning: no previous prototype for function 'determine_component_type' [-Wmissing-prototypes]
std::string determine_component_type(const char* name) {
            ^
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:47:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
std::string determine_component_type(const char* name) {
^
static 
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:92:6: warning: no previous prototype for function 'read_gguf_string' [-Wmissing-prototypes]
bool read_gguf_string(FILE* f, std::string& out) {
     ^
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:92:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
bool read_gguf_string(FILE* f, std::string& out) {
^
static 
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:110:6: warning: no previous prototype for function 'skip_gguf_value' [-Wmissing-prototypes]
bool skip_gguf_value(FILE* f, uint32_t type) {
     ^
/Users/ersibesi/Desktop/LLAMA/llama.cpp/tools/gguf-dump/gguf-dump.cpp:110:1: note: declare 'static' if the function is not intended to be used outside of this translation unit
bool skip_gguf_value(FILE* f, uint32_t type) {
^
static 
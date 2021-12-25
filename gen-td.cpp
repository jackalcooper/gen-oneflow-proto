#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>

using namespace google::protobuf::compiler;
using namespace google::protobuf;
class MyCodeGenerator : public CodeGenerator {
public:
  MyCodeGenerator();
  MyCodeGenerator(MyCodeGenerator &&) = default;
  MyCodeGenerator(const MyCodeGenerator &) = default;
  MyCodeGenerator &operator=(MyCodeGenerator &&) = default;
  MyCodeGenerator &operator=(const MyCodeGenerator &) = default;
  ~MyCodeGenerator();

  bool Generate(const FileDescriptor *file, const std::string &parameter,
                GeneratorContext *generator_context,
                std::string *error) const override {
    return true;
  }
};

MyCodeGenerator::MyCodeGenerator() {}

MyCodeGenerator::~MyCodeGenerator() {}

int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}

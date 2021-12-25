#include <google/protobuf/compiler/code_generator.h>
#include <google/protobuf/compiler/plugin.h>
#include <google/protobuf/descriptor.h>

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
                std::string *error) const override;
};

MyCodeGenerator::MyCodeGenerator() {}

MyCodeGenerator::~MyCodeGenerator() {}

bool MyCodeGenerator::Generate(const FileDescriptor *file,
                               const std::string &parameter,
                               GeneratorContext *generator_context,
                               std::string *error) const {
  std::cerr << parameter << "\n";
  std::cerr << *error << "\n";
  std::cerr << file->name() << "\n";

  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}

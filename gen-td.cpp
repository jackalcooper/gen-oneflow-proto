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

#define FOR_RANGE(i, end) for (size_t i = (0), __end = (end); i < __end; ++i)
#define LOG(x) std::cerr << x << "\n";
namespace {
void ConverFields(const google::protobuf::Descriptor *d) {
  FOR_RANGE(i, d->field_count()) {
    auto f = d->field(i);
    auto t = f->message_type();
    std::cerr << "  - ";
    if (t) {
      std::cerr << t->name() << ": ";
    }
    LOG(f->name());
  }
}

const int USER_OP_NUMBER = 199;
bool IsSystemOp(int number) { return number > 100 && number < USER_OP_NUMBER; }

} // namespace

bool MyCodeGenerator::Generate(const FileDescriptor *file,
                               const std::string &parameter,
                               GeneratorContext *generator_context,
                               std::string *error) const {
  std::cerr << parameter << "\n";
  std::cerr << *error << "\n";
  std::cerr << file->name() << "\n";
  // for (int i; i < file->message_type_count(); i++) {
  //   auto m = file->message_type(i);
  //   std::cerr << m->name() << "\n";
  // }
  auto OperatorConf = file->FindMessageTypeByName("OperatorConf");
  std::cerr << OperatorConf->name() << "\n";
  auto op_type = OperatorConf->FindOneofByName("op_type");
  for (int filed_i; filed_i < op_type->field_count(); filed_i += 1) {
    auto m = op_type->field(filed_i);
    const bool should_convert = IsSystemOp(m->number());
    if (!should_convert)
      continue;
    std::cerr << m->number() << "\n";
    std::cerr << m->name() << "\n";
    ConverFields(m->message_type());
  }
  return true;
}
int main(int argc, char **argv) {
  MyCodeGenerator generator;
  return google::protobuf::compiler::PluginMain(argc, argv, &generator);
}

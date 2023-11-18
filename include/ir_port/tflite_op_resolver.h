#pragma once

#include <map>
#include <vector>

#include "common/logging.h"
#include "common/stl_wrapper.h"
#include "model/operator.h"
#include "model/options.h"
#include "model/types.h"
#include "schema_generated.h"

class BaseOptionResolver {
 public:
    virtual ~BaseOptionResolver() = default;

    virtual void SerializeOption(const Operator&                   op,
                                 ::flatbuffers::FlatBufferBuilder* builder) const = 0;

    virtual void ParseOption(const void* builtin_options,
                             Operator&                       op) const = 0;
};

template <typename Tp, typename Up>
class TfLiteOptionResolver : public BaseOptionResolver {
 public:
    using TfLiteOptionT = Tp;
    using BaseOptionT   = Up;

    void SerializeOption(const Operator& op, ::flatbuffers::FlatBufferBuilder* builder) const {
        // SerializeOptionImpl();
    }

    void ParseOption(const void* builtin_options, Operator& op) const {
        auto* option        = op.GetOption<BaseOptionT>();
        auto* tflite_option = static_cast<const TfLiteOptionT*>(builtin_options);
        ParseOptionImpl(*option, *tflite_option);
    }

 protected:
    virtual void ParseOptionImpl(BaseOptionT&, const TfLiteOptionT&) const                  = 0;
    virtual void SerializeOptionImpl(const BaseOptionT&, ::flatbuffers::FlatBufferBuilder*) const = 0;
};

class OperatorResolver {
 public:
    OperatorResolver();

    template <typename T, typename = std::enable_if_t<std::is_base_of_v<BaseOptionResolver, T>>>
    void AddOpResolver(OperatorType op_type, ::tflite::BuiltinOperator tflite_type) {
        option_resolver_map_[op_type] = std::make_unique<T>();
        op_type_table_.push_back({op_type, tflite_type});
    }

    OperatorType GetMappedOpTypeOf(::tflite::BuiltinOperator op_code) const {
        auto target = common::find_if(
            op_type_table_, [&](const auto& type_pair) { return type_pair.second == op_code; });
        return target->first;
    }
    ::tflite::BuiltinOperator GetMappedOpTypeOf(OperatorType op_type) const {
        auto target = common::find_if(
            op_type_table_, [&](const auto& type_pair) { return type_pair.first == op_type; });
        return target->second;
    }

    const BaseOptionResolver* GetOptionResolver(OperatorType op_type) const {
        return option_resolver_map_.at(op_type).get();
    }

    ~OperatorResolver() = default;

 private:
    using OpResolverMap  = std::map<OperatorType, std::unique_ptr<BaseOptionResolver>>;
    using OpTypeMapTable = std::vector<std::pair<OperatorType, ::tflite::BuiltinOperator>>;
    OpResolverMap  option_resolver_map_;
    OpTypeMapTable op_type_table_;
};

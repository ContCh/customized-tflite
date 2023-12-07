#pragma once

#include <atomic>

#include "common/singleton.h"

#define GenUniqueID() IDGenerator::GetInstance()->GenID()
#define ResetGenID()  IDGenerator::GetInstance()->Reset()

typedef uint32_t GEN_ID_TYPE;

class IDGenerator : public Singleton<IDGenerator> {
 public:
    GEN_ID_TYPE GenID() { return global_id_++; }

    void Reset() { global_id_ = 0; }

    friend class Singleton<IDGenerator>;

 private:
    IDGenerator()                       = default;
    std::atomic<GEN_ID_TYPE> global_id_ = 0;
};

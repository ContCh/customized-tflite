#pragma once

#include <memory>
#include <mutex>

template <typename T>
class Singleton {
 public:
    static T* GetInstance() {
        if (instance_ == nullptr) {
            mtx_.lock();
            if (instance_ == nullptr) {
                instance_ = std::unique_ptr<T>(new T());
            }
            mtx_.unlock();
        }
        return instance_.get();
    }

    virtual ~Singleton() {};

 protected:
    Singleton()                 = default;
    Singleton(const Singleton&) = default;
    Singleton& operator=(const Singleton&) = default;

    static std::unique_ptr<T> instance_;
    static std::mutex mtx_;
};

template <typename T>
std::unique_ptr<T> Singleton<T>::instance_ = nullptr;

template <typename T>
std::mutex Singleton<T>::mtx_;

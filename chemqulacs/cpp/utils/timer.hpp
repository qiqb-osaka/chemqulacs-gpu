/**
 * @file timer.hpp
 * @brief Header file for Timer classes
 * @author Yusuke Teranishi
 */
#pragma once

#include <cuda_runtime.h>

#include <cassert>
#include <chrono>
#include <queue>
#include <string>
#include <unordered_map>

/**
 * @brief Timer class
 */
class Timer {
   protected:
    bool _is_record;
    double _sum_time;

   public:
    Timer() : _is_record(false), _sum_time(0) {}
    virtual ~Timer() {}

    virtual void start() = 0;
    virtual void stop() = 0;
    virtual void reset() = 0;
    virtual void restart() = 0;
    virtual double getTime() = 0;
    virtual double getTimeRestart() = 0;
    virtual bool isRecord() { return _is_record; }
};

/**
 * @brief TimerHost class
 * - Timer class for host
 */
class TimerHost : public Timer {
   private:
    std::chrono::system_clock::time_point _t_start;
    std::chrono::system_clock::time_point _tnd;

   public:
    /**
     * @brief Start timer
     */
    void start() {
        assert(!_is_record);
        _is_record = true;
        _t_start = std::chrono::system_clock::now();
    }
    /**
     * @brief Stop timer
     */
    void stop() {
        assert(_is_record);
        _is_record = false;
        _tnd = std::chrono::system_clock::now();
        double t =
            static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(_tnd - _t_start).count() / 1e6);
        _sum_time += t;
    }
    /**
     * @brief Reset timer
     */
    void reset() {
        _is_record = false;
        _sum_time = 0;
    }
    /**
     * @brief Restart timer
     */
    void restart() {
        reset();
        start();
    }

    /**
     * @brief Get total time
     */
    double getTime() {
        assert(!_is_record);
        return _sum_time;
    }
    /**
     * @brief Get total time and restart timer
     */
    double getTimeRestart() {
        stop();
        double t = getTime();
        restart();
        return t;
    }
};

/**
 * @brief TimerDevice class
 */
class TimerDevice : public Timer {
   private:
    class CudaEventPair {
       public:
        cudaEvent_t start;
        cudaEvent_t stop;

        CudaEventPair() {
            cudaEventCreate(&start);
            cudaEventCreate(&stop);
        }
        ~CudaEventPair() {
            cudaEventDestroy(start);
            cudaEventDestroy(stop);
        }
    };
    std::queue<CudaEventPair*> _used_queue;
    std::queue<CudaEventPair*> _available_queue;

    /**
     * @brief Consume events
     */
    void _consume(bool is_sync) {
        float milliseconds;
        while (!_used_queue.empty()) {
            CudaEventPair* cep = _used_queue.front();
            if (cudaEventQuery(cep->stop) != cudaSuccess) {
                if (is_sync) {
                    cudaEventSynchronize(cep->stop);
                } else {
                    break;
                }
            }
            cudaEventElapsedTime(&milliseconds, cep->start, cep->stop);
            _sum_time += (double)milliseconds / 1e3;
            _used_queue.pop();
            _available_queue.push(cep);
        }
    }

   public:
    /**
     * @brief Constructor
     */
    TimerDevice() { _available_queue.push(new CudaEventPair()); }

    /**
     * @brief Destructor
     */
    ~TimerDevice() {
        while (!_available_queue.empty()) {
            CudaEventPair* cep = _available_queue.front();
            _available_queue.pop();
            delete cep;
        }
        while (!_used_queue.empty()) {
            CudaEventPair* cep = _used_queue.front();
            _used_queue.pop();
            delete cep;
        }
    }

    /**
     * @brief Start timer
     */
    void start() {
        assert(!_is_record);
        _is_record = true;

        // consume
        _consume(false);

        // produce
        if (_available_queue.empty()) {
            _available_queue.push(new CudaEventPair());
        }

        CudaEventPair* cep = _available_queue.front();
        cudaEventRecord(cep->start);
    }

    /**
     * @brief Stop timer
     */
    void stop() {
        assert(_is_record);
        _is_record = false;

        assert(!_available_queue.empty());
        CudaEventPair* cep = _available_queue.front();
        _available_queue.pop();
        _used_queue.push(cep);
        cudaEventRecord(cep->stop);
    }

    /**
     * @brief Reset timer
     */
    void reset() {
        _is_record = false;
        _sum_time = 0;
        while (!_used_queue.empty()) {
            CudaEventPair* cep = _used_queue.front();
            _used_queue.pop();
            _available_queue.push(cep);
        }
    }

    /**
     * @brief Restart timer
     */
    void restart() {
        reset();
        start();
    }

    /**
     * @brief Get total time
     */
    double getTime() {
        assert(!_is_record);
        _consume(true);
        return _sum_time;
    }

    /**
     * @brief Get total time and restart timer
     */
    double getTimeRestart() {
        stop();
        double t = getTime();
        restart();
        return t;
    }
};

/**
 * @brief TimerDict class
 */
class TimerDict {
   private:
    std::unordered_map<std::string, Timer*> _timer_dict;
    std::unordered_map<std::string, double> _register_time;
    std::unordered_map<std::string, double> _time_dict;

   public:
    /**
     * @brief Destructor
     */
    ~TimerDict() {
        for (const auto & [ name, timer ] : _timer_dict) {
            delete timer;
        }
    }

    /**
     * @brief Add timer for host
     * @param[in] name Name of the timer
     * @return Timer object
     */
    Timer* addTimerHost(std::string name) {
        Timer* timer = new TimerHost();
        _timer_dict[name] = timer;
        return timer;
    }

    /**
     * @brief Add timer for device
     * @param[in] name Name of the timer
     * @return Timer object
     */
    Timer* addTimerDevice(std::string name) {
        Timer* timer = new TimerDevice();
        _timer_dict[name] = timer;
        return timer;
    }

    /**
     * @brief Add timer for host
     * @param[in] name Name of the timer
     * @return Timer object
     */
    Timer* addTimer(std::string name) { return addTimerHost(name); }

    /**
     * @brief Get timer
     */
    Timer* getTimer(std::string name) { return _timer_dict[name]; }

    /**
     * @brief Register time
     * @param[in] name Name of the timer
     * @param[in] time Time
     */
    void registerTime(std::string name, double time) { _register_time[name] = time; }

    /**
     * @brief Get time dictionary
     * @return Time dictionary
     */
    std::unordered_map<std::string, double> getTimeDict() {
        for (const auto & [ name, timer ] : _timer_dict) {
            _time_dict[name] = timer->getTime();
        }
        for (const auto & [ name, time ] : _register_time) {
            _time_dict[name] = time;
        }
        return _time_dict;
    }
};

extern TimerDict _TimerDict;
extern TimerDevice _timer_update_compute;
extern TimerDevice _timer_update_communicate;
extern TimerDevice _timer_exp_compute;
extern TimerDevice _timer_exp_communicate;

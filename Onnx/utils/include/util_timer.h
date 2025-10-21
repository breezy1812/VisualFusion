#ifndef INCLUDE_CORE_TIMER_H
#define INCLUDE_CORE_TIMER_H

#include <iostream>

#include <string>
#include <chrono>

namespace core
{

  class Timer
  {
  public:
    explicit Timer(std::string name);

    void start();
    void stop();
    void show();

  private:
    std::string name;

    int count = 0;
    double total_time = 0;

    std::chrono::time_point<std::chrono::system_clock> start_time;
  };

} 

#endif 

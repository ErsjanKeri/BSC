#include <sys/mman.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>


void signalHandler( int signum ) {
   std::cout << "Interrupt signal (" << signum << ") received.\n";
   exit(signum);
}

int main(int argc, char** argv) {
   using namespace std::chrono_literals;

   signal(SIGINT, signalHandler);

   size_t len = atoi(argv[1]) * (1024ull*1024*1024);
   std::cout << mlockall(MCL_FUTURE) << std::endl;

   void* p = malloc(len);
   memset(p, 0, len);
   std::cout << len << " " << p << std::endl;

   std::cout << "..." << std::endl;
   while (true) {
      std::this_thread::sleep_for(2000ms);
   }

   return 0;
}

#include <sys/mman.h>
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <csignal>
#include <chrono>
#include <thread>
#include <fstream>


void signalHandler( int signum ) {
   std::cout << "Interrupt signal (" << signum << ") received.\n";
   exit(signum);
}

int main(int argc, char** argv) {
   using namespace std::chrono_literals;

   if (argc < 2) {
      std::cerr << "Usage: " << argv[0] << " <size_in_GiB>" << std::endl;
      return 1;
   }

   signal(SIGINT, signalHandler);

   // Make ourselves immune to OOM killer (requires root)
   std::ofstream oom_adj("/proc/self/oom_score_adj");
   if (oom_adj.is_open()) {
      oom_adj << "-1000" << std::endl;
      oom_adj.close();
      std::cout << "OOM score adj set to -1000 (immune to OOM killer)" << std::endl;
   } else {
      std::cerr << "Warning: could not set oom_score_adj (run as root)" << std::endl;
   }

   size_t len = atoi(argv[1]) * (1024ull*1024*1024);
   std::cout << "Locking " << (len / (1024*1024*1024)) << " GiB..." << std::endl;
   std::cout << "mlockall result: " << mlockall(MCL_FUTURE) << std::endl;

   void* p = malloc(len);
   memset(p, 0, len);
   std::cout << "Locked " << len << " bytes at " << p << " (PID " << getpid() << ")" << std::endl;

   std::cout << "Press Ctrl+C to release and exit." << std::endl;
   while (true) {
      std::this_thread::sleep_for(2000ms);
   }

   return 0;
}

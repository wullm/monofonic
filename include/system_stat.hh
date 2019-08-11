#pragma once

#include <string>

#ifdef __APPLE__
#include <sys/types.h>
#include <sys/sysctl.h>
#include <mach/mach.h>
#include <mach/vm_statistics.h>
#include <mach/mach_types.h>
#include <mach/mach_init.h>
#include <mach/mach_host.h>
#include <unistd.h>
#elif __linux__
#include <cstring>
#include <cstdio>
#include <strings.h>
#include <libcpuid.h>
#endif

namespace SystemStat
{

class Cpu
{
public:
    Cpu() {}

    std::string get_CPUstring() const
    {
#ifdef __APPLE__
        char buffer[1024];
        size_t size=sizeof(buffer);
        if (sysctlbyname("machdep.cpu.brand_string", &buffer, &size, NULL, 0) < 0) {
            return "";
        }
        return std::string(buffer);
#elif __linux__
        std::string str = "";
        FILE *cpuinfo = fopen("/proc/cpuinfo", "rb");
        char *arg = 0;
        size_t size = 0;
        while(getdelim(&arg, &size, '\n', cpuinfo) != -1)
        {
            if( strncmp(arg, "model name", 10) == 0 ){
                str = std::string(arg+13);
                break;
            }
        }
        free(arg);
        fclose(cpuinfo);
        return str;
#endif
    }
};

class Memory
{
private:
    size_t total;
    size_t avail;
    size_t used;

public:
    Memory()
        : total(0), avail(0), used(0)
    {
        this->get_statistics();
    }

    size_t get_TotalMem() const { return this->total; }
    size_t get_AvailMem() const { return this->avail; }
    size_t get_UsedMem() const { return this->used; }
    void update() { this->get_statistics(); }

protected:
    int get_statistics(void)
    {
#ifdef __APPLE__
        int64_t pagesize = int64_t(getpagesize());
        int mib[2] = {CTL_HW, HW_MEMSIZE};
        size_t length = sizeof(size_t);
        sysctl(mib, 2, &this->total, &length, nullptr, 0);

        vm_statistics64 vmstat;
        natural_t mcount = HOST_VM_INFO64_COUNT;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&vmstat), &mcount) == KERN_SUCCESS)
        {
#if 1 // count inactive as available
            this->avail = (int64_t(vmstat.free_count) +
                           int64_t(vmstat.inactive_count)) *
                          pagesize;
            this->used = (int64_t(vmstat.active_count) +
                          int64_t(vmstat.wire_count)) *
                         pagesize;
#else // count inactive as unavailable
            this->avail = int64_t(vmstat.free_count) * pagesize;
            this->used = (int64_t(vmstat.active_count) +
                          int64_t(vmstat.inactive_count) +
                          int64_t(vmstat.wire_count)) *
                         pagesize;
#endif
        }

#elif __linux__
        FILE *fd;
        char buf[1024];
        if ((fd = fopen("/proc/meminfo", "r")))
        {
            while (1)
            {
                if (fgets(buf, 500, fd) != buf)
                    break;
                if (bcmp(buf, "MemTotal", 8) == 0)
                {
                    this->total = atoll(buf + 10) * 1024; // in Mb
                }
                if (strncmp(buf, "Committed_AS", 12) == 0)
                {
                    this->used = atoll(buf + 14) * 1024; // in Mb
                }
                // if(strncmp(buf, "SwapTotal", 9) == 0)
                // {
                //     *SwapTotal = atoll(buf + 11);
                // }
                // if(strncmp(buf, "SwapFree", 8) == 0)
                // {
                //     *SwapFree = atoll(buf + 10);
                // }
            }
            fclose(fd);
        }
        this->avail = this->total - this->used;

#endif
        return 0;
    }
};

} /* namespace SystemStat */

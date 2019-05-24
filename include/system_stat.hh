#pragma once

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
#endif


namespace SystemStat {
    
    class Memory{
    private:
        size_t	total;
        size_t	avail;
        size_t	used;
    public:
        Memory() 
        : total(0), avail(0), used(0) 
        {
            this->get_statistics();
        }

        size_t get_TotalMem() const { return this->total; }
        size_t get_AvailMem() const { return this->avail; }
        size_t get_UsedMem() const { return this->used; }
        void update(){ this->get_statistics(); }

    protected:
        int get_statistics( void )
        {
        #ifdef __APPLE__
            int pagesize = getpagesize();
            int mib[2] = {CTL_HW, HW_MEMSIZE};
            size_t length = sizeof(size_t);
            sysctl(mib, 2, &this->total, &length, nullptr, 0);

            mach_msg_type_number_t count = HOST_VM_INFO_COUNT;
            vm_statistics_data_t vmstat;
            if(KERN_SUCCESS != host_statistics(mach_host_self(), HOST_VM_INFO, (host_info_t)&vmstat, &count)) {
                return -2;
            }
            this->avail = (int64_t)vmstat.free_count * (int64_t)pagesize;
            this->used = ((int64_t)vmstat.active_count +
                        (int64_t)vmstat.inactive_count +
                        (int64_t)vmstat.wire_count) *  (int64_t)pagesize;

        #elif __linux__
            FILE *fd;
            char buf[1024];
            if((fd = fopen("/proc/meminfo", "r")))
            {
                while(1)
                {
                    if(fgets(buf, 500, fd) != buf) break;
                    if(bcmp(buf, "MemTotal", 8) == 0)
                    {
                        this->total = atoll(buf + 10);
                    }
                    if(strncmp(buf, "Committed_AS", 12) == 0)
                    {
                        this->used = atoll(buf + 14);
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

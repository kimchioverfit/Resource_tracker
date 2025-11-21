// main.cpp  - PC365 시스템 성능 CSV 로거
// 빌드 타겟 예: PC365_Logger.exe
// 사용법 예:   PC365_Logger.exe PC365_log_#11_A.csv 1000

#include <windows.h>
#include <string>
#include <chrono>
#include <thread>
#include <atomic>
#include <vector>
#include <iostream>
#include <algorithm>
#include "pdh_helper.h"     // CPU, Memory, Disk, Network PDH 카운터 래퍼
#include "wmi_helper.h"     // CPU 모델명, 코어수, RAM 등 정적 정보
#include "disk_helper.h"    // 디스크 이름 / 타입 등
#include "net_helper.h"     // NIC 이름 등
#include "csv_writer.h"     // CSVWriter: writeHeader(), writeRow()
#include "util_format.h"    // formatDouble(), formatInt() 등 헬퍼
#include "nvml_helper.h"    // NVML 기반 GPU 정보 (없으면 더미 구현해도 됨)

#pragma comment(lib, "wbemuuid.lib")

static std::atomic_bool g_stop{false};

static BOOL WINAPI ConsoleCtrlHandler(DWORD ctrlType) {
    switch (ctrlType) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
    case CTRL_CLOSE_EVENT:
        g_stop = true;
        return TRUE;
    default:
        return FALSE;
    }
}

// ISO 시간 문자열 (로컬타임)
static std::string nowIso() {
    SYSTEMTIME st{};
    GetLocalTime(&st);
    char buf[64];
    // "YYYY-MM-DDThh:mm:ss.mmm"
    sprintf_s(buf, "%04d-%02d-%02dT%02d:%02d:%02d.%03d",
        st.wYear, st.wMonth, st.wDay,
        st.wHour, st.wMinute, st.wSecond, st.wMilliseconds);
    return std::string(buf);
}

int main(int argc, char* argv[])
{
    std::string outCsv = "PC365_log.csv";
    int interval_ms = 1000; // 기본 1초

    if (argc >= 2) {
        outCsv = argv[1];
    }
    if (argc >= 3) {
        interval_ms = (std::max)(100, atoi(argv[2]));
    }

    std::cout << "[PC365] output = " << outCsv << ", interval_ms = " << interval_ms << "\n";

    // 콘솔 Ctrl+C 핸들러
    SetConsoleCtrlHandler(ConsoleCtrlHandler, TRUE);

    try {
        // =========================
        // 1. 정적 정보 (WMI)
        // =========================
        WmiHelper wmi;
        auto cpuInfo  = wmi.queryCpuInfo();   // 모델명, 코어/스레드, 기본클럭
        auto memInfo  = wmi.queryMemoryInfo(); // 전체 RAM(MB)

        DiskHelper diskHelper;
        auto diskInfo = diskHelper.querySystemDiskInfo();
        
        NetHelper netHelper;
        auto netInfo = netHelper.queryPrimaryNicInfo();

        // =========================
        // 2. PDH / NVML 초기화
        // =========================
        PdhHelper pdh;
        pdh.init(); // 내부에서 Query, Counter 등록 (CPU, MEM, DISK, NET, Cache, Paging, …)

        NvmlHelper nvml;
        bool haveGpu = nvml.init(); // NVML 초기화 실패해도 false로 처리

        // 첫 샘플 버퍼링용 warm-up
        pdh.collect();

        // =========================
        // 3. CSV 헤더 작성
        // =========================
        CsvWriter csv(outCsv, /*append=*/false);

        std::vector<std::string> header = {
            "Time",
            "CPU Model",
            "CPU Cores(Physical)",
            "CPU Threads(Logical)",
            "CPU Base(MHz)",
            "Total RAM(MB)",

            // CPU
            "CPU Utilization(%)",
            "CPU Speed(MHz)",
            "CPU Temperature(C)",
            "CPU QueueLen",
            "CPU CtxSwitches(/sec)",
            "% of Max CPU Freq",
            "% Privileged Time",
            "% User Time",
            "DPCs/sec",
            "Interrupts/sec",

            // Memory
            "MEM Utilization(%)",
            "MEM Committed(MB)",
            "MEM PageFaults(/sec)",
            "Available MBytes",
            "Free&Zero List Bytes(MB)",
            "Standby Reserve Bytes(MB)",
            "Pool Nonpaged Bytes(MB)",
            "Pool Paged Bytes(MB)",
            "Cache Bytes(MB)",
            "Cache Faults/sec",
            "Pages/sec",
            "Page Reads/sec",
            "Page Writes/sec",
            "Pages Input/sec",
            "Transition Faults/sec",
            "Commit Limit(MB)",
            "%Committed Bytes In Use",
            "PagingFile %Usage",
            "PagingFile %Usage Peak",

            // Disk (전체 디스크 또는 특정 인스턴스, 예: _Total 또는 PhysicalDrive0)
            "DISK I/O (%)",
            "DISK AvgSec/Read",
            "DISK AvgSec/Write",
            "DISK CurrentQueueLen",
            "DISK AvgBytes/Transfer",
            "DISK SplitIO/sec",
            "DISK AvgReadQLen",
            "DISK AvgWriteQLen",
            "DISK Transfers/sec",
            "DISK %IdleTime",
            "DISK ReadBytes/sec",
            "DISK WriteBytes/sec",

            // Network
            "NET BytesRecv/sec",
            "NET BytesSent/sec",
            "NET BytesTotal/sec",

            // GPU (NVML 사용)
            "GPU Name",
            "GPU Temperature(C)",
            "GPU Utilization(%)",
            "GPU MemUtilization(%)",
            "GPU MemUsed(MB)",
            "GPU MemTotal(MB)",
            "GPU Power(W)",
            "GPU PCIE Rx(KB/s)",
            "GPU PCIE Tx(KB/s)",

            "__source_file"
        };

        csv.writeHeader(header);

        std::cout << "[PC365] logging started. Press Ctrl+C to stop.\n";

        // =========================
        // 4. 메인 루프
        // =========================
        while (!g_stop.load()) {
            auto start = std::chrono::steady_clock::now();

            // PDH 수집 (이전 샘플 대비 delta 기반인 카운터들 있음)
            pdh.collect();

            // ========== CPU ==========
            double cpuUtil     = pdh.getCpuTotalUtil();
            double cpuSpeedMHz = pdh.getCpuSpeedMHz();
            double cpuTempC    = pdh.getCpuTemperature();
            double cpuQueueLen = pdh.getCpuQueueLength();
            double cpuCtxSw    = pdh.getCpuContextSwitchesPerSec();
            double cpuMaxFreq  = pdh.getCpuPercentOfMaxFreq();
            double cpuPrivPct  = pdh.getCpuPrivilegedTime();
            double cpuUserPct  = pdh.getCpuUserTime();
            double dpcsPerSec  = pdh.getDpcsPerSec();
            double intrPerSec  = pdh.getInterruptsPerSec();

            // ========== MEM ==========
            double memUtilPct      = pdh.getMemUtilPercent();
            double memCommittedMB  = pdh.getMemCommittedMB();
            double memPageFaults   = pdh.getMemPageFaultsPerSec();
            double memAvailMB      = pdh.getMemAvailableMB();
            double memFreeZeroMB   = pdh.getMemFreeAndZeroMB();
            double memStandbyMB    = pdh.getMemStandbyReserveMB();
            double memNonPagedMB   = pdh.getMemPoolNonPagedMB();
            double memPagedMB      = pdh.getMemPoolPagedMB();
            double cacheBytesMB    = pdh.getCacheBytesMB();
            double cacheFaults     = pdh.getCacheFaultsPerSec();
            double pagesPerSec     = pdh.getPagesPerSec();
            double pageReadsPerSec = pdh.getPageReadsPerSec();
            double pageWritesPerSec= pdh.getPageWritesPerSec();
            double pagesInputPerSec= pdh.getPagesInputPerSec();
            double transitionFaults= pdh.getTransitionFaultsPerSec();
            double commitLimitMB   = pdh.getCommitLimitMB();
            double pctCommitInUse  = pdh.getPercentCommittedBytesInUse();
            double pfUsage         = pdh.getPagingFileUsagePercent();
            double pfUsagePeak     = pdh.getPagingFileUsagePeakPercent();

            // ========== DISK ==========
            double diskBusyPct      = pdh.getDiskPercentDiskTime();
            double diskAvgSecRead   = pdh.getDiskAvgSecPerRead();
            double diskAvgSecWrite  = pdh.getDiskAvgSecPerWrite();
            double diskCurQueueLen  = pdh.getDiskCurrentQueueLength();
            double diskAvgBytesXfer = pdh.getDiskAvgBytesPerTransfer();
            double diskSplitIO      = pdh.getDiskSplitIOPerSec();
            double diskAvgReadQLen  = pdh.getDiskAvgReadQueueLen();
            double diskAvgWriteQLen = pdh.getDiskAvgWriteQueueLen();
            double diskXferPerSec   = pdh.getDiskTransfersPerSec();
            double diskIdlePct      = pdh.getDiskIdleTimePercent();
            double diskReadBps      = pdh.getDiskReadBytesPerSec();
            double diskWriteBps     = pdh.getDiskWriteBytesPerSec();

            // ========== NET ==========
            double netRecvBps  = pdh.getNetBytesRecvPerSec();
            double netSentBps  = pdh.getNetBytesSentPerSec();
            double netTotalBps = netRecvBps + netSentBps;

            // ========== GPU (NVML) ==========
            std::string gpuName;
            double gpuTempC       = 0.0;
            double gpuUtilPct     = 0.0;
            double gpuMemUtilPct  = 0.0;
            double gpuMemUsedMB   = 0.0;
            double gpuMemTotalMB  = 0.0;
            double gpuPowerW      = 0.0;
            double gpuPcieRxKBs   = 0.0;
            double gpuPcieTxKBs   = 0.0;

            if (haveGpu) {
                NvmlGpuMetrics gm{};
                if (nvml.queryFirstGpu(gm)) {
                    gpuName      = gm.name;
                    gpuTempC     = gm.temperatureC;
                    gpuUtilPct   = gm.gpuUtilPercent;
                    gpuMemUtilPct= gm.memUtilPercent;
                    gpuMemUsedMB = gm.memUsedMB;
                    gpuMemTotalMB= gm.memTotalMB;
                    gpuPowerW    = gm.powerDrawW;
                    gpuPcieRxKBs = gm.pcieRxKBs;
                    gpuPcieTxKBs = gm.pcieTxKBs;
                }
            }

            // ========== CSV 한 줄 작성 ==========
            std::vector<std::string> row;
            row.reserve(header.size());

            row.push_back(nowIso());
            row.push_back(cpuInfo.modelName);
            row.push_back(std::to_string(cpuInfo.physicalCores));
            row.push_back(std::to_string(cpuInfo.logicalProcessors));
            row.push_back(util::formatDouble(cpuInfo.baseClockMHz, 1));
            row.push_back(std::to_string(memInfo.totalRamMB));

            row.push_back(util::formatDouble(cpuUtil, 2));
            row.push_back(util::formatDouble(cpuSpeedMHz, 1));
            row.push_back(util::formatDouble(cpuTempC, 1));
            row.push_back(util::formatDouble(cpuQueueLen, 2));
            row.push_back(util::formatDouble(cpuCtxSw, 2));
            row.push_back(util::formatDouble(cpuMaxFreq, 2));
            row.push_back(util::formatDouble(cpuPrivPct, 2));
            row.push_back(util::formatDouble(cpuUserPct, 2));
            row.push_back(util::formatDouble(dpcsPerSec, 2));
            row.push_back(util::formatDouble(intrPerSec, 2));

            row.push_back(util::formatDouble(memUtilPct, 2));
            row.push_back(util::formatDouble(memCommittedMB, 2));
            row.push_back(util::formatDouble(memPageFaults, 2));
            row.push_back(util::formatDouble(memAvailMB, 2));
            row.push_back(util::formatDouble(memFreeZeroMB, 2));
            row.push_back(util::formatDouble(memStandbyMB, 2));
            row.push_back(util::formatDouble(memNonPagedMB, 2));
            row.push_back(util::formatDouble(memPagedMB, 2));
            row.push_back(util::formatDouble(cacheBytesMB, 2));
            row.push_back(util::formatDouble(cacheFaults, 2));
            row.push_back(util::formatDouble(pagesPerSec, 2));
            row.push_back(util::formatDouble(pageReadsPerSec, 2));
            row.push_back(util::formatDouble(pageWritesPerSec, 2));
            row.push_back(util::formatDouble(pagesInputPerSec, 2));
            row.push_back(util::formatDouble(transitionFaults, 2));
            row.push_back(util::formatDouble(commitLimitMB, 2));
            row.push_back(util::formatDouble(pctCommitInUse, 2));
            row.push_back(util::formatDouble(pfUsage, 2));
            row.push_back(util::formatDouble(pfUsagePeak, 2));

            row.push_back(util::formatDouble(diskBusyPct, 2));
            row.push_back(util::formatDouble(diskAvgSecRead, 4));
            row.push_back(util::formatDouble(diskAvgSecWrite, 4));
            row.push_back(util::formatDouble(diskCurQueueLen, 2));
            row.push_back(util::formatDouble(diskAvgBytesXfer, 2));
            row.push_back(util::formatDouble(diskSplitIO, 2));
            row.push_back(util::formatDouble(diskAvgReadQLen, 2));
            row.push_back(util::formatDouble(diskAvgWriteQLen, 2));
            row.push_back(util::formatDouble(diskXferPerSec, 2));
            row.push_back(util::formatDouble(diskIdlePct, 2));
            row.push_back(util::formatDouble(diskReadBps, 2));
            row.push_back(util::formatDouble(diskWriteBps, 2));

            row.push_back(util::formatDouble(netRecvBps, 2));
            row.push_back(util::formatDouble(netSentBps, 2));
            row.push_back(util::formatDouble(netTotalBps, 2));

            row.push_back(gpuName);
            row.push_back(util::formatDouble(gpuTempC, 1));
            row.push_back(util::formatDouble(gpuUtilPct, 1));
            row.push_back(util::formatDouble(gpuMemUtilPct, 1));
            row.push_back(util::formatDouble(gpuMemUsedMB, 1));
            row.push_back(util::formatDouble(gpuMemTotalMB, 1));
            row.push_back(util::formatDouble(gpuPowerW, 1));
            row.push_back(util::formatDouble(gpuPcieRxKBs, 1));
            row.push_back(util::formatDouble(gpuPcieTxKBs, 1));

            row.push_back(outCsv); // __source_file

            csv.writeRow(row);

            // ========== 샘플링 인터벌 맞추기 ==========
            auto end   = std::chrono::steady_clock::now();
            auto spent = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            int sleep_ms = interval_ms - static_cast<int>(spent);
            if (sleep_ms > 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(sleep_ms));
        }

        std::cout << "[PC365] logging stopped.\n";
    }
    catch (const std::exception& ex) {
        std::cerr << "[PC365] exception: " << ex.what() << "\n";
        return 1;
    }

    return 0;
}

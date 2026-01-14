#include "cpu_info.h"

#include <sstream>

#if defined(_MSC_VER) && defined(_MANAGED)
#using <System.dll>
#using <System.Core.dll>
#using <LibreHardwareMonitorLib.dll>
#include <msclr/marshal_cppstd.h>
using namespace System;
using namespace LibreHardwareMonitor::Hardware;
#endif

std::string get_cpu_info_lhm() {
#if defined(_MSC_VER) && defined(_MANAGED)
    try {
        Computer^ computer = gcnew Computer();
        computer->IsCpuEnabled = true;
        computer->Open();

        String^ cpu_name;
        double cpu_temp_c = -1.0;
        double cpu_load_pct = -1.0;

        for each (IHardware^ hw in computer->Hardware) {
            if (hw->HardwareType != HardwareType::CPU) continue;
            cpu_name = hw->Name;
            hw->Update();
            for each (ISensor^ sensor in hw->Sensors) {
                if (!sensor->Value.HasValue) continue;
                if (sensor->SensorType == SensorType::Temperature && cpu_temp_c < 0.0) {
                    cpu_temp_c = sensor->Value.Value;
                } else if (sensor->SensorType == SensorType::Load &&
                           sensor->Name->Contains("CPU Total") && cpu_load_pct < 0.0) {
                    cpu_load_pct = sensor->Value.Value;
                }
            }
        }

        computer->Close();

        std::ostringstream ss;
        ss << "cpu_name=";
        if (cpu_name) {
            ss << msclr::interop::marshal_as<std::string>(cpu_name);
        }
        ss << ", cpu_temp_c=" << cpu_temp_c << ", cpu_load_pct=" << cpu_load_pct;
        return ss.str();
    } catch (Exception^ ex) {
        return "LibreHardwareMonitor error: " +
               msclr::interop::marshal_as<std::string>(ex->Message);
    }
#else
    return "LibreHardwareMonitor requires MSVC C++/CLI (/clr).";
#endif
}

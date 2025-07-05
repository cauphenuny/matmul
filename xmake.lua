add_rules("mode.debug", "mode.release")
add_rules("plugin.compile_commands.autoupdate")

add_requires("pybind11", {system = false, configs = {python = "python3"}})
add_requires("openmp")

-- 设置详细输出模式
set_policy("build.warning", true)
set_policy("build.optimization.lto", false)
set_warnings("all")

-- 在调试模式下添加更多调试信息
if is_mode("debug") then
    set_symbols("debug")
    set_optimize("none")
    add_defines("DEBUG", "_DEBUG")
    add_cxflags("-g", "-O0", "-fno-omit-frame-pointer")
    add_ldflags("-g")
    -- 添加地址消毒器（可选）
    -- add_cxflags("-fsanitize=address")
    -- add_ldflags("-fsanitize=address")
end

-- 在发布模式下的优化设置
if is_mode("release") then
    set_symbols("hidden")
    set_optimize("fastest")
    add_defines("NDEBUG")
    set_strip("all")
end

target("matmul")
    set_kind("shared")
    set_extension(".so")
    set_languages("c99", "c++17")
    add_files("src/*.cpp")
    add_packages("pybind11")
    add_cxxflags("-fopenmp")
    add_ldflags("-fopenmp")
    add_shflags("-fopenmp")
    set_policy("build.warning", true)
    
    -- 添加系统 Python 配置
    add_rules("python.library", {soabi = true})
    
    -- 添加详细的编译信息
    on_load(function (target)
        print("Loading target: " .. target:name())
        print("Build mode: " .. get_config("mode"))
        print("Architecture: " .. get_config("arch"))
        print("Platform: " .. get_config("plat"))
    end)
    
    before_build(function (target)
        print("Starting build for target: " .. target:name())
        print("Source files:")
        for _, file in ipairs(target:sourcefiles()) do
            print("  - " .. file)
        end
    end)
    
    after_build(function (target)
        local file = target:targetfile()
        local linkfile = path.join(os.projectdir(), "benchmark", "libmatmul.so")
        
        print("Build completed successfully!")
        print("Target file: " .. file)
        print("File size: " .. os.filesize(file) .. " bytes")
        
        if os.exists(linkfile) then
            os.rm(linkfile)
            print("Removed existing symlink: " .. linkfile)
        end
        
        os.cp(file, linkfile)
        print("Copied to benchmark directory: " .. linkfile)
        
        -- 显示编译统计信息
        print("Build statistics:")
        print("  - Build directory: " .. target:targetdir())
        print("  - Object files: " .. #target:objectfiles())
        
        -- 验证生成的库文件
        if os.exists(linkfile) then
            print("✓ Shared library is ready at: " .. linkfile)
        else
            print("✗ Failed to create shared library at: " .. linkfile)
        end
    end)

-- 添加自定义任务以显示详细信息
task("info")
    on_run(function ()
        print("=== Project Information ===")
        print("Project directory: " .. os.projectdir())
        print("Build directory: " .. path.join(os.projectdir(), "build"))
        print("Current mode: " .. get_config("mode"))
        print("Current arch: " .. get_config("arch"))
        print("Current plat: " .. get_config("plat"))
        print("Compiler: " .. get_config("cc"))
        print("Toolchain: " .. get_config("toolchain"))
        
        -- 显示目标信息
        for _, target in pairs(project.targets()) do
            print("\n=== Target: " .. target:name() .. " ===")
            print("Kind: " .. target:kind())
            print("Languages: " .. target:get("languages"))
            print("Files: " .. target:get("files"))
            print("Packages: " .. target:get("packages"))
        end
    end)
    set_menu {
        usage = "xmake info",
        description = "Show detailed project information",
    }

-- If you want to known more usage about xmake, please see https://xmake.io
--
-- ## FAQ
--
-- You can enter the project directory firstly before building project.
--
--   $ cd projectdir
--
-- 1. How to build project?
--
--   $ xmake
--
-- 2. How to configure project?
--
--   $ xmake f -p [macosx|linux|iphoneos ..] -a [x86_64|i386|arm64 ..] -m [debug|release]
--
-- 3. Where is the build output directory?
--
--   The default output directory is `./build` and you can configure the output directory.
--
--   $ xmake f -o outputdir
--   $ xmake
--
-- 4. How to run and debug target after building project?
--
--   $ xmake run [targetname]
--   $ xmake run -d [targetname]
--
-- 5. How to install target to the system directory or other output directory?
--
--   $ xmake install
--   $ xmake install -o installdir
--
-- 6. Add some frequently-used compilation flags in xmake.lua
--
-- @code
--    -- add debug and release modes
--    add_rules("mode.debug", "mode.release")
--
--    -- add macro definition
--    add_defines("NDEBUG", "_GNU_SOURCE=1")
--
--    -- set warning all as error
--    set_warnings("all", "error")
--
--    -- set language: c99, c++11
--    set_languages("c99", "c++11")
--
--    -- set optimization: none, faster, fastest, smallest
--    set_optimize("fastest")
--
--    -- add include search directories
--    add_includedirs("/usr/include", "/usr/local/include")
--
--    -- add link libraries and search directories
--    add_links("tbox")
--    add_linkdirs("/usr/local/lib", "/usr/lib")
--
--    -- add system link libraries
--    add_syslinks("z", "pthread")
--
--    -- add compilation and link flags
--    add_cxflags("-stdnolib", "-fno-strict-aliasing")
--    add_ldflags("-L/usr/local/lib", "-lpthread", {force = true})
--
-- @endcode
--


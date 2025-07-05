add_rules("mode.debug", "mode.release")
-- add_rules("plugin.compile_commands.autoupdate")

add_requires("pybind11", {system = false, configs = {python = "python3"}})
add_requires("python 3.x", {system = false})
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
    add_packages("python")
    add_cxxflags("-fopenmp")
    add_ldflags("-fopenmp")
    add_shflags("-fopenmp")
    set_policy("build.warning", true)
    
    -- 添加系统 Python 配置
    add_rules("python.library", {soabi = true})
    
    -- 添加详细的编译信息
    on_load(function (target)
        print("Loading target: " .. target:name())
        print("Build mode: " .. (get_config("mode") or "release"))
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

        print("Running uv sync...")
        os.exec("uv sync")
        
        print("Generating Python stubs...")
        local projectdir = os.projectdir()
        local stub_script = path.join(projectdir, "scripts", "generate_stubs.py")
        os.exec("uv run " .. stub_script)
        print("✓ Python stubs generated successfully")
        
    end)

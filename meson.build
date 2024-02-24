project('mat', 'cpp',
  version : '0.1',
  default_options : ['warning_level=3'])

exe = executable('main', 'cmd/main.cpp',
  install : true,
  cpp_args: '-std=c++23')

test('basic', exe)

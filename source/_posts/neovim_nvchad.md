---
title: Neovim预配置库NvChad探索
tags: [vim, IDE]
categories: coding 
date: 2022-8-11
---

# 简介
`vim`是一个非常强大的文本编辑器，而`Neovim`是对原`vim`的一个分叉，不过最简单的`Neovim`显得太过朴素，不过它提供了强大的插件系统，可以通过各种插件将其由一个简约的文本编辑器转化为强大的代码开发IDE。
[`NvChad`](https://nvchad.github.io/)就是一个`Neovim`的预配置库，可以使得`Neovim`开箱即获得各种强大的功能：
- `NvChad`是一个用`lua`编写的`neovim`配置，旨在提供一个具有非常漂亮的用户界面和极快的启动时间的基本配置（在基础硬件上约0.02秒至0.07秒）；
- 懒加载的机制使得插件不会被默认加载，只有在需要的时候才会被加载，包括特定的命令和vim事件等。这有助于减少启动时间，从而使启动时间比平时快；
- `NvChad`不是一个框架，它是作为大众的 "基础 "配置使用的。它的目的是提供一个特定的默认插件集。

# 安装
## 前提条件
在使用`NvChad`之前，要有一些前提依赖：
- `Neovim 0.7.2`及以上
安装教程在[这里](https://github.com/neovim/neovim/wiki/Installing-Neovim)。
- 字体及图标：
一个推荐的编程字体是`Fira Code`字体，这里不光安装它，还安装它的一个扩展，即`Nerd fonts`（`Nerd fonts` 本身并不是一种新的字体，而是把常用图标以打补丁的方式打到了常用字体上。）
具体到[官网这里](https://www.nerdfonts.com/font-downloads)进行下载。
对于`Linux`字体的安装，步骤为：
```sh
sudo unzip FiraCode.zip -d /usr/share/fonts
sudo fc-cache -fv
````
对于`Windows`版本，注意在下载的文件中选择`XXXX Windows Compatible.ttf`。然后在`Windows Terminal`的字体中选择`FiraCode NF`字体即可。）
为了测试是否成功，可以到[这个网址](www.nerdfonts.com/cheat-sheet)，点击 `Show All Icons` 按钮，选择一个图标，点击右上角的 `Copy Icon`，然后粘贴到命令行里即可。
- 保证所需的目录干净
对于Linux和MacOS系统，删除`~/.local/share/nvim`这个文件夹；对于Windows系统，删除`~\AppData\Local\nvim`和`~\AppData\Local\nvim-data`。
- 如果要用到`Telescope`的模糊搜索，还要提前安装`ripgrep`。具体安装方法可以参考官方文档[BurntSushi/ripgrep](https://github.com/BurntSushi/ripgrep)。

## 安装
对于Linux/MacOS系统：
```sh
git clone https://github.com/NvChad/NvChad ~/.config/nvim --depth 1 && nvim
```
对于Windows系统：
（注意：还需提前安装[`mingw`](https://www.mingw-w64.org/)，以及在环境变量中配置其路径）
```sh
git clone https://github.com/NvChad/NvChad $HOME\AppData\Local\nvim --depth 1 && nvim
```

## 升级
`NvChad`有一个内置的升级机制，快捷键是`<leader> + uu`。
注意，默认配置下`<leader>`键是空格键`<space>`。
具体地，它在后台会使用`git reset --hard`来获取官方git库中的更新，因此在`lua/custom`文件夹外的改动都会被丢弃（因此，如果想在`NvChad`上自定义配置，需要在`lua/custom`文件夹内进行）。

## 卸载
```sh
# linux/macos (unix)
rm -rf ~/.config/nvim
rm -rf ~/.local/share/nvim
rm -rf ~/.cache/nvim

# windows
rd -r ~\AppData\Local\nvim
rd -r ~\AppData\Local\nvim-data
```

# 安装之后
上一步安装好`NvChad`后，还可以做一些进阶动作，当然不做也不影响使用。

## 创建自定义配置
- `NvChad`的更新不会覆盖`lua/custom`目录，因为它被`gitignored`了，因此所有的用户修改都必须在这个文件夹中完成。
- `lua/custom/init.lua`在`init.lua`主文件的最后被加载，可以在这里添加自定义的命令等。
- `lua/custom/chadrc.lua`用于覆盖`lua/core/default_config.lua`并基本上控制整个`nvchad`，因此必须采用跟`default_config.lua`一样的代码结构。

`NvChad`在`examples`文件夹中提供了`init.lua`和`chadrc.lua`，可以将其作为默认初始的自定义配置文件，将其复制到`custom`文件夹中：
```sh
mkdir lua/custom
cp examples/init.lua lua/custom/init.lua
cp examples/chadrc.lua lua/custom/chadrc.lua
```

## 安装Treesitter解析器
[`nvim-treesitter`](https://github.com/nvim-treesitter/nvim-treesitter)提供了代码高亮、缩进和折叠等功能。
`nvim-treesitter`的正常运行需要满足以下条件：
- 在环境变量中能找到`tar`和`curl`命令（或者`git`命令）
- 在环境变量中有`C`编译器和`libstdc++`
在`Linux`系统中，使用`sudo apt install build-essential`即可安装相应依赖，在`Windows`系统中，可查看[该详细教程](https://github.com/nvim-treesitter/nvim-treesitter/wiki/Windows-support)


安装解析器使用以下命令（以`Python`为例）：
```sh
:TSInstall python
```

## 安装Node.JS
`Node.js`对于后面的`LSP`是有用的，比如安装`pyright`时需要用到`npm`，所以这里也可以事先安装。
安装包在[这里](https://nodejs.org/zh-cn/)。
对于`Ubuntu`等`Linux`系统，也可以使用包管理器来安装，教程见[这里](https://nodejs.org/zh-cn/download/package-manager/)。
以`Ubuntu`为例：
```sh
curl -fsSL https://deb.nodesource.com/setup_16.x | sudo -E bash -
sudo apt-get install -y nodejs
```


# Lua简明教程
## Print
```lua
print("Hi")
```
## 注释
```lua
-- comment
print("Hi") -- comment

--[[
 multi-line 
 comment
]]
```
## 变量
```lua
-- Different types

local x = 10 -- number
local name = "sid" -- string
local isAlive = true -- boolean
local a = nil --no value or invalid value
```

## 条件表达式
```lua
-- Number comparisons
local age = 10

if age > 18 then
  print("over 18") -- this will not be executed
end

-- elseif and else
age = 20

if age > 18 then
  print("over 18")
elseif age == 18 then
  print("18 huh")
else
  print("kiddo")
end
```
```lua
-- Boolean comparison
local isAlive = true

if isAlive then
    print("Be grateful!")
end

-- String comparisons
local name = "sid"

if name ~= "sid" then
  print("not sid")
end
```
### 组合表达式
```lua
local age = 22

if age == 10 and x > 0 then -- both should be true
  print("kiddo!")
elseif x == 18 or x > 18 then -- 1 or more are true
  print("over 18")
end

-- result: over 18
```

### 反转
```lua
local x = 18

if not x == 18 then
  print("kiddo!") -- prints nothing as x is 18
end
```

## 函数
```lua
local function print_num(a)
  print(a)
end

or

local print_num = function(a)
  print(a)
end

print_num(5) -- prints 5 
```
```lua
-- multiple parameters

function sum(a, b)
  return a + b
end
```
## 作用域
```lua
function foo()
  local n = 10
end

print(n) -- nil , n isn't accessible outside foo()
```
## 循环
### While
```lua
local i = 1

while i <= 3 do
   print("hi")
   i = i + 1
end
```
### For
```lua
for i = 1, 3 do
   print("hi")
   i = i + 1
end
```

## Tables
### 数组（列表）
```lua
local colors = { "red", "green", "blue" }

print(colors[1]) -- red
print(colors[2]) -- green
print(colors[3]) -- blue

-- Different ways to loop through lists
-- #colors is the length of the table, #tablename is the syntax

for i = 1, #colors do
  print(colors[i])
end

-- ipairs 
for index, value in ipairs(colors) do
   print(colors[index])
   -- or
   print(value)
end

-- If you dont use index or value here then you can replace it with _ 
for _, value in ipairs(colors) do
   print(value)
end
```
### 字典
```lua
local info = { 
   name = "sid",
   age = 20,
   isAlive = true
}

-- both print sid
prrint(info["name"])
print(info.name)

-- Loop by pairs
for key, value in pairs(info) do
   print(key .. " " .. tostring(value))
end

-- prints name sid, age 20 etc
```

### 嵌套Tables
```lua
-- Nested lists
local data = {
    { "Sid", 20 },
    { "Tim", 90 },
}

for i = 1, #data do
  print(data[i][1] .. " is " .. data[i][2] .. " years old")
end

-- Nested dictionaries
local data = {
    sid = { age = 20 },
    time = { age = 90 },
}
```

## 模块
```lua
require("path")
```

## Neovim中的Lua应用
Neovim中的配置文件和插件都可以使用`Lua`进行编写，具体的教程，可以参考[该文档](https://github.com/nanotee/nvim-lua-guide)。

# 配置
## 总览
`NvChad`的文件结构如下：
```lua
├── init.lua
│
├── lua
│   │
│   ├── core
│   │   ├── default_config.lua
│   │   ├── mappings.lua
│   │   ├── options.lua
│   │   ├── packer.lua  -- (bootstrap packer & installs plugins)
│   │   ├── utils.lua  -- (core util functions) (i)
│   │   └── init.lua  -- (autocmds)
│   │
│   ├── plugins
│   │    ├── init.lua -- (default plugin list)
│   │    └── configs
│   │        ├── cmp.lua
│   │        ├── others.lu -- (list of small configs of plugins)
│   │        └── many more plugin configs
|   |
│   ├── custom *
│   │   ├── chadrc.lua -- (overrides default_config)
│   │   ├── init.lua -- (runs after main init.lua file)
│   │   ├── more files, dirs
```

### 配置主题
快捷键是：`<leader> + th`。

### 快捷键映射
`:Telescope keymaps`

### 默认设置
默认设置在`lua/core/default_config.lua`。

## 选项
在`custom/init.lua`中可以进行如下操作：
- 重载默认选项
- 设置`autocmds`和`global`全局变量
- 懒加载，具体可以查看[packer readme](https://github.com/wbthomason/packer.nvim#specifying-plugins)
- 代码片段，比如：`vim.g.luasnippets_path = "your snippets path"`

## 插件
`NvChad`在底层使用`packer.nvim`，不过它重新定义了一套语法，比如：
- `packer.nvim`定义插件的方式：
```lua
   use { "NvChad/nvterm" }, -- without any options

   -- with more options
   use {
      "NvChad/nvterm"
      module = "nvterm",
      config = function()
         require "plugins.configs.nvterm"
      end,
   },
```
- `NvChad`定义插件的方式：
```lua
   ["NvChad/nvterm"] = {}, -- without any options

   -- with more options
   ["NvChad/nvterm"] = {
      module = "nvterm",
      config = function()
         require "plugins.configs.nvterm"
      end,
   },
```

### 安装插件
首先创建`lua/custom/plugins/init.lua`，按以下格式添加插件：
```lua
-- custom/plugins/init.lua has to return a table
-- THe plugin name is github user or organization name/reponame

return {

   ["elkowar/yuck.vim"] = { ft = "yuck" },

   ["max397574/better-escape.nvim"] = {
      event = "InsertEnter",
      config = function()
         require("better_escape").setup()
      end,
   },
}
```
然后在`lua/custom/chadrc.lua`中引入该文件（实际写完一次后就不用更改了）：
```lua
-- chadrc.lua

M.plugins = {
   user = require "custom.plugins"
}
```
最后执行`:PackerSync`即可。

### 重载插件的默认配置
当想从一个插件的默认配置选项中改变一个东西，但又不想复制粘贴整个配置时，这个功能就很有用了。
如下：
```lua
M.plugins = {
   override = {
      ["nvim-treesitter/nvim-treesitter"] = {
        ensure_installed = {
          "html",
          "css",
       },
     }
   }
}
```
但以上面这种方式的话，配置一多了就会显得很乱，可以采用如下这种方式：
```lua
local pluginConfs = require "custom.plugins.configs"

M.plugins = {
   override = {
      ["nvim-treesitter/nvim-treesitter"] = pluginConfs.treesitter,
      ["kyazdani42/nvim-tree.lua"] = pluginConfs.nvimtree,
   },
}
```
```lua
-- custom/plugins/configs.lua file

local M = {}

M.treesitter = {
   ensure_installed = {
      "lua",
      "html",
      "css",
   },
}

M.nvimtree = {
   git = {
      enable = true,
   },
   view = {
      side = "right",
      width = 20,
   },
}

-- you cant directly call a module in chadrc thats related to the default config 
-- Thats because most probably that module is lazyloaded
-- In this case its 'cmp', we have lazyloaded it by default
-- So you need to make this override field a function, instead of a table 
-- And the function needs to return a table!

M.cmp = function()
   local cmp = require 'cmp' 

   return {
      mapping = {
         ["<C-d>"] = cmp.mapping.scroll_docs(-8),
      }
    }
end

return M
```
然后执行`:PackerSync`。

### 修改插件的配置
比如在`lua/custom/plugins/init.lua`中是这样定义`nvimtree`的：
```lua
 ["kyazdani42/nvim-tree.lua"] = {
      cmd = { "NvimTreeToggle", "NvimTreeFocus" },

      setup = function()
         require("core.mappings").nvimtree()
      end,

      config = function()
         require "plugins.configs.nvimtree"
      end,
 }
```
现在想修改其中的`config`和`cmd`配置，那么可以保持该文件不动，在安装它们时再改：
```lua
M.plugins = {
  user = {
      ["kyazdani42/nvim-tree.lua"] = {
      cmd = { "abc" },
      config = function()
          require "custom.plugins.nvimtree"
      end,
   }
} 

-- This will change cmd, config values from default plugin definition
-- So the setup value isnt changed, look close!
```

### 删除插件
```lua
M.plugins = {
  remove = {
      "andymass/vim-matchup",
      "NvChad/nvterm",
   },
}

-- now run :PackerSync
```

## 快捷键映射
- `C`是`Ctrl`
- `<leader>`是`<space>`
- `A`是`Alt`
- `S`是`Shift`
- 默认配置在`core/mappings.lua`中定义。

### 快捷键映射格式
```lua
-- opts here is completely optional

 ["keys"] = {"action", "icon  mapping description", opts = {}},

 -- more examples
 ["<C-n>"] = {"<cmd> NvimTreeToggle <CR>", "Toggle nvimtree", opts = {}},

 ["<leader>uu"] = { "<cmd> :NvChadUpdate <CR>", "  update nvchad" },

 [";"] = { ":", "enter cmdline", opts = { nowait = true } },
 ["jk"] = { "<ESC>", "escape insert mode" , opts = { nowait = true }},

 -- example with lua function
 ["<leader>tt"] = {
     function()
        require("base46").toggle_theme()
     end,
        "   toggle theme",
   },
```
- 映射描述对`Whichkey`是必要的，对于非`Whichkey`则不是必需
- 可以使用图标来帮助阅读，不过也不是必选项，而是可选项
- 可以从[这里](https://www.nerdfonts.com/cheat-sheet)来复制和粘贴图标
- 默认的`opts`的值有：
```lua
{
  buffer = nil, -- Global mappings. Specify a buffer number for buffer local mappings
  silent = true, 
  noremap = true,
  nowait = false,

  -- all standard key binding opts are supported 
}
```

### 增加新的映射
默认配置在`core/mappings.lua`中定义，然后可以在`custom/mappings.lua`中增加新的映射：
```lua
-- lua/custom/mappings 
local M = {}

-- add this table only when you want to disable default keys
M.disabled = {
  n = {
      ["<leader>h"] = "",
      ["<C-s>"] = ""
  }
}

M.abc = {

  n = {
     ["<C-n>"] = {"<cmd> Telescope <CR>", "Open Telescope"}
  }

  i = {
    -- more keys!
  }
}

M.xyz = {
  -- stuff
}

return M
```
注意在自己的`custom/chadrc.lua`中引入：
```lua
-- chadrc
M.mappings = require "custom.mappings"
```
- 上面的`abc`和`xyz`是随意取的，为了可读性，可以改成插件名字
- 上面的映射关系是自动加载的，不需要手动加载。



## UI插件
`NvChad`使用了自己的[UI插件](https://github.com/NvChad/ui)。

## LSP
以下出在[掘金上的这个小册](https://juejin.cn/book/7051157342770954277)：
> 想要在 Neovim 中配置代码补全、代码悬停、代码提示等等功能，首先要了解什么是 LSP (Language Server Protocol) 语言服务协议。
在 LSP 出现之前，传统的 IDE 都要为其支持的每个语言实现类似的代码补全、文档提示、跳转到定义等功能，不同的 IDE 做了很多重复的工作，并且兼容性也不是很好。 LSP 的出现将编程工具解耦成了 Language Server 与 Language Client 两部分。定义了编辑器与语言服务器之间交互协议。
Client 专注于显示样式实现， Server 负责提供语言支持，包括常见的自动补全、跳转到定义、查找引用、悬停文档提示等功能。
而这里所说的 Neovim 内置 LSP 就是说 Neovim 内置了一套 Language Client 端的实现，这样就可以连接到和 VSCode 相同的第三方 language servers ，实现高质量的语法补全等功能。

还有一个简单的教程见[这里](http://xfyuan.github.io/2021/02/neovim-builtin-lsp-basic-configuration/)。
> 为了简化 LSP 的安装和配置，NeoVim 官方专门创建了 `nvim-lspconfig` 插件来帮助我们。这个插件把所有 LSP 背后的繁琐都封装到其内部，让使用者再也毋需担心出现费了大半天功夫结果仍然无法用起来的事。

### 搭建内部LSP
```lua
-- we are just modifying lspconfig's packer definition table
-- put this in your custom plugins section i.e M.plugins.user field 

["neovim/nvim-lspconfig"] = {
    config = function()
      require "plugins.configs.lspconfig"
      require "custom.plugins.lspconfig"
    end,
},
```

```lua
-- custom.plugins.lspconfig
local on_attach = require("plugins.configs.lspconfig").on_attach
local capabilities = require("plugins.configs.lspconfig").capabilities

local lspconfig = require "lspconfig"
local servers = { "html", "cssls", "clangd", "pyright"}

for _, lsp in ipairs(servers) do
  lspconfig[lsp].setup {
    on_attach = on_attach,
    capabilities = capabilities,
  }
end
```

然后执行`:PackerCompile`。

### 外部LSP server
可以使用`Mason.nvim`来安装外部LSP Server。
输入命令`:Mason`，就能打开它的浮动窗口，来安装、更新、卸载相关的安装包（比如lspservers、linters和formatters等）。
最好是使用配置文件来使用`Mason.nvim`，如：
```lua
 ["williamboman/mason.nvim"] = {
      ensure_installed = {
        -- lua stuff
        "lua-language-server",
        "stylua",

        -- web dev
        "css-lsp",
        "html-lsp",
        "typescript-language-server",
        "deno",
        "emmet-ls",
        "json-lsp",

        -- shell
        "shfmt",
        "shellcheck",
      },
    },
```
然后执行`:MasonInstallAll`（注意该命令是`NvChad`的定制命令，不是官方原有命令）。


## DAP
使用`Debug Adapter Protocol`（`DAP`）需要安装一各特定语言的`Debug Adapter`。
`python`的话就是使用`debugpy`。
### 使用conda安装debugpy
正常安装`miniconda`或者`ananconda`后，激活某个虚拟环境，然后安装：
```python
pip install debugpy
```
然后在`custom/plugins/dap/dap-python.lua`中使用`nvim-dap-python`的默认配置即可：
```lua
require('dap-python').setup('python')
```

### 在venv中debugpy
```sh
sudo apt install python3.10-venv
python -m venv path/to/virtualenvs/debugpy
path/to/virtualenvs/debugpy/bin/python -m pip install debugpy
```
然后在`custom/plugins/dap/dap-python.lua`中将该路径配置进去：
```lua
require('dap-python').setup('path/to/virtualenvs/debugpy/bin/python')
```
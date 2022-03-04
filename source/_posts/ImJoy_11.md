---
title: 开源深度学习计算平台ImJoy解析：11 -- ImJoy主站之主组件
tags: [ImJoy]
categories: computer vision 
date: 2022-2-28
---

# 简介
前面讲了[ImJoy主站](https://imjoy.io)的入口文件`main.js`，这一篇解析一下该主站的ImJoy组件，它也是整个app的“门面担当”，起到了统筹整个网站的作用。

为了便于分析，将很多高阶的暂时用不到的组件（比如工具栏、窗口管理、文件上传、连接后台Engine等）都删掉，留下最基本的能运行最小化imjoy插件的功能，截图如下：
![mini-imjoy](https://user-images.githubusercontent.com/6218739/156502647-18321726-6c25-4631-bcc4-b632eef39f23.png)

这样便于分析整个组件的宏观结构和运行逻辑。
下面是对于该极小化组件的逐行代码分析。
# template代码
```js
<template>
  <div class="imjoy noselect">
    <!-- 整个imjoy页面框架是通过vue material这个组件库的md-app进行组织的 -->
    <!-- 它包括了md-app-toolbar工具栏、 md-app-drawer菜单栏和md-app-content内容区三部分 -->
    <!-- 相关教程见：https://www.creative-tim.com/vuematerial/components/app -->
    <md-app>
      <!-- 工具栏就是最上面的横条，里面的内容直接全部删除了，仅保留架子 -->
      <md-app-toolbar class="md-dense app-toolbar" md-elevation="0">
      </md-app-toolbar>

      <!-- 接下来就是左侧的菜单栏 -->
      <!-- drawer的属性设置见：https://www.creative-tim.com/vuematerial/components/drawer# -->
      <!-- 具体地： -->
      <!-- (1)菜单栏是否可见是通过md-active属性控制，其通过v-bind绑定到了menuVisible这个变量上 -->
      <!-- (2)菜单栏关闭和打开两个事件都通过v-on绑定到了wm.resizeAll()函数上 -->
      <!-- (3)菜单栏是否常驻md-persistent和菜单栏是否支持触屏下的swipe绑定到了screenWidth这个变量上 -->
      <md-app-drawer
        :md-active.sync="menuVisible"
        @md-closed="wm.resizeAll()"
        @md-opened="wm.resizeAll()"
        :md-persistent="screenWidth > 800 ? 'full' : null"
        :md-swipeable="screenWidth > 600 ? false : true"
      >
        <!-- 接下来是将上传文件,添加工作流,添加插件和插件列表都放在了一个card中 -->
        <!-- 这个card的显示有两个判断语句,一个是用v-show判断plugin_loaded这个变量是否为true -->
        <!-- 另一个是用v-if判断pm这个变量是否为true, -->
        <!-- v-if和v-show看起来差不多,但有区别,见:https://cn.vuejs.org/v2/guide/conditional.html -->
        <md-card id="plugin-menu" v-show="plugin_loaded" v-if="pm">
          <!-- 将files,workflow和plugins都放在了card的header中, -->
          <!-- 这里我们将files和workflow都删掉了,仅留plugins的代码 -->
          <md-card-header>
            <!-- 添加plugins就是一个按钮,其class也是由v-bind绑定并判断 -->
            <!-- 该按钮的点击时间用v-on绑定到了showPluginManagement()函数上 -->
            <!-- 当点击该按钮后, 从而执行该函数,其中有一个非常重要的变量:-->
            <!-- this.showAddPluginDialog会设为true -->
            <!-- 从而会引发如下对话框的弹出:
            <md-dialog
              class="plugin-dialog"
              :md-active.sync="showAddPluginDialog"
              :md-click-outside-to-close="true"
            > -->
            <md-button
              ref="add_plugin_button"
              :class="pm.installed_plugins.length > 0 ? '' : 'md-primary'"
              @click="showPluginManagement()"
            >
              <md-icon>add</md-icon>Plugins
            </md-button>
          </md-card-header>

          <!-- 接下来就是将插件列表放在card的content区域 -->
          <md-card-content>
            <!-- 使用v-for来循环插件列表,并对每一项提供了key的身份 -->
            <!-- v-for的教程见:https://cn.vuejs.org/v2/guide/list.html -->
            <div v-for="plugin in sortedRunnablePlugins()" :key="plugin.name">
              <!-- 增加一个分割线 -->
              <md-divider></md-divider>
              <div style="display: flex;">
                <!-- 这里对每一个插件会包装一个badge徽章,用途是判断它需不需要升级 -->
                <!-- 如果需要升级,会出现一个NEW的角标 -->
                <md-badge
                  :class="plugin.update_available ? '' : 'hide-badge'"
                  class="md-square md-primary"
                  md-dense
                  md-content="NEW"
                >
                  <!-- 这里是对插件的icon按钮点击后会出现的菜单进行定义 -->
                  <md-menu md-size="medium">
                    <!-- 触发菜单的icon按钮定义 -->
                    <md-button
                      class="md-icon-button"
                      :class="plugin.running ? 'md-accent' : ''"
                      md-menu-trigger
                    >
                      <!-- 插件的加载状态 -->
                      <md-progress-spinner
                        v-if="plugin.initializing || plugin.terminating"
                        class="md-accent"
                        :md-diameter="20"
                        md-mode="indeterminate"
                      ></md-progress-spinner>
                      <!-- icon会判断该插件有没有提供自定义的图标,如果没有,则使用默认的extension图标 -->
                      <!-- 这个地方调用的就是之前入口文件所全局注册的PluginIcon组件 -->
                      <plugin-icon
                        v-else
                        :icon="plugin.config.icon"
                      ></plugin-icon>
                      <!-- 按钮的工具提示信息 -->
                      <md-tooltip v-if="screenWidth > 500">{{
                        plugin.name + ": " + plugin.config.description
                      }}</md-tooltip>
                    </md-button>

                    <!-- 点击icon按钮后弹出的菜单选项 -->
                    <md-menu-content>
                      <!-- Docs选项,绑定了showDoc方法 -->
                      <md-menu-item @click="showDoc(plugin.id)">
                        <md-icon>description</md-icon>Docs
                      </md-menu-item>
                      <!-- Share选项,不一定会显示,有一个v-if条件渲染 -->
                      <md-menu-item
                        v-if="plugin.config.origin"
                        @click="sharePlugin(plugin.id)"
                      >
                        <md-icon>share</md-icon>Share
                      </md-menu-item>
                      <!-- Export选项,绑定了downloadPlugin -->
                      <md-menu-item @click="downloadPlugin(plugin.id)">
                        <md-icon>cloud_download</md-icon>Export
                      </md-menu-item>
                      <!-- Edit选项,绑定了editPlugin -->
                      <md-menu-item @click="editPlugin(plugin.id)">
                        <md-icon>edit</md-icon>Edit
                      </md-menu-item>
                      <!-- Reload选项,绑定了reloadPlugin -->
                      <md-menu-item @click="reloadPlugin(plugin.config)">
                        <md-icon>autorenew</md-icon>Reload
                      </md-menu-item>
                      <!-- Terminate选项,绑定了unloadPlugin -->
                      <md-menu-item @click="unloadPlugin(plugin)">
                        <md-icon>clear</md-icon>Terminate
                      </md-menu-item>
                      <!-- Remove选项,绑定了removePlugin -->
                      <md-menu-item
                        class="md-accent"
                        @click="removePlugin(plugin)"
                      >
                        <md-icon>delete_forever</md-icon>Remove
                      </md-menu-item>

                    </md-menu-content>
                  </md-menu>
                </md-badge>

                <!-- 接下来就是由插件的名称所形成的按钮 -->
                <!-- 对于需要链接到engine的插件,还会检测其状态,如果没有链接到engine上,就会disable该按钮,从而无法运行. -->
                <!-- 对于鼠标操作,也附加了鼠标按钮修饰符,防止误操作,教程见:https://cn.vuejs.org/v2/guide/events.html -->
                <!-- 点击该按钮后就会运行runOp方法 -->
                <md-button
                  class="joy-run-button"
                  :class="
                    plugin.running
                      ? 'busy-plugin'
                      : plugin._disconnected && plugin.engine
                      ? 'md-accent'
                      : 'md-primary'
                  "
                  :disabled="plugin._disconnected && !plugin.engine"
                  @click.exact="
                    plugin._disconnected
                      ? connectPlugin(plugin)
                      : runOp(plugin.ops[plugin.name])
                  "
                  @click.right.exact="logPlugin(plugin)"
                >
                  {{ plugin.config.name + " " + plugin.config.badges }}
                </md-button>

              </div>

              <!-- 这个地方是ImJoy的核心!! -->
              <!-- 此处也是ImJoy的名称的来源,即ImJoy来自于joy.js这个库 -->
              <!-- 这个地方需要后面仔细研究机理 -->
              <div
                v-for="op in plugin.ops"
                :key="op.plugin_id + op.name"
                v-show="true"
              >
                <joy :config="op" :show="true"></joy>
              </div>
              <md-divider></md-divider>
            </div>
            <md-divider></md-divider>
          </md-card-content>
        </md-card>
      </md-app-drawer>

      <!-- 接下来就是整个界面的第三部分,即内容区 -->
      <md-app-content
        :class="workspace_dropping ? 'file-dropping' : ''"
        class="whiteboard-content"
      >
        <!-- 就是引用了whiteboard这个组件 -->
        <whiteboard
          v-if="wm"
          id="whiteboard"
          @create="createWindow($event)"
          :mode="wm.window_mode"
          :window-manager="wm"
        ></whiteboard>
      </md-app-content>

    <!-- 以上就是整个界面的布局 -->
    </md-app>

    <!-- 这个是消息提醒对话框，当某处调用showAlert()函数时会触发，比如api.alert时-->
    <md-dialog-alert
      class="api-dialog"
      :md-active.sync="alert_config.show"
      :md-title="alert_config.title"
      :md-content="alert_config.content"
      :md-confirm-text="alert_config.confirm_text"
    />

    <!-- 这个是消息确认对话框，当某处调用showConfirm()函数时会触发，比如删除插件时 -->
    <md-dialog-confirm
      class="api-dialog"
      :md-active.sync="confirm_config.show"
      :md-title="confirm_config.title"
      :md-content="confirm_config.content"
      :md-confirm-text="confirm_config.confirm_text"
      :md-cancel-text="confirm_config.canel_text"
      @md-confirm="confirm_config.confirm"
      @md-cancel="confirm_config.cancel"
    />

    <!-- 前面已分析到，点击添加Plugins后会触发如下插件对话框的弹出 -->
    <md-dialog
      class="plugin-dialog"
      :md-active.sync="showAddPluginDialog"
      :md-click-outside-to-close="true"
    >
      <!-- 设置该对话框的标题 -->
      <md-dialog-title
        >{{
          plugin4install ? "Plugin Installation" : "ImJoy Plugin Management"
        }}
        <!-- 在标题栏添加一个关闭按钮 -->
        <md-button
          class="md-accent"
          style="position:absolute; top:8px; right:5px;"
          @click="
            showAddPluginDialog = false;
            clearPluginUrl();
          "
          ><md-icon>clear</md-icon></md-button
        ></md-dialog-title
      >
      <!-- 以下是对话框的内容部分 -->
      <!-- 原来的插件对话框有三种添加插件的方式：（1）通过插件模板template编写插件（2）输入插件url地址安装（3）通过插件商店store安装 -->
      <!-- 这里我们为了代码精简，只保留了第一种，保证能运行一个最简单的插件即可 -->
      <md-dialog-content>
        <template v-if="show_plugin_templates">
          <md-menu>
            <!-- 通过模板template添加插件的按钮 -->
            <md-button class="md-primary md-raised" md-menu-trigger>
              <md-icon>add</md-icon>Create a new plugin
              <md-tooltip>Create a new plugin</md-tooltip>
            </md-button>
            <!-- 点击上面按钮后弹出的菜单 -->
            <md-menu-content>
              <!-- 菜单项会对所有的模板进行遍历展示 -->
              <!-- 每个菜单项也都会将鼠标点击事件绑定到newPlugin方法上 -->
              <!-- 该方法会通过createWindow创建plugin-editor类型的窗口，即代码编辑器 -->
              <md-menu-item
                @click="
                  newPlugin(template.code);
                  showAddPluginDialog = false;
                "
                v-for="template in plugin_templates"
                :key="template.name"
              >
                <md-icon>{{ template.icon }}</md-icon
                >{{ template.name }}
              </md-menu-item>
            </md-menu-content>
          </md-menu>
          <br />
          <br />
        </template>

      </md-dialog-content>
    </md-dialog>
  </div>
</template>
```

# script代码
通过上述template文件的解析，发现插件被点击运行时执行的函数是runOp方法。
该代码解析留坑待填。

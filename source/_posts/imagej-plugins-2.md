---
title: ImageJ的插件开发(2)
tags: [ImageJ]
categories: programming
date: 2018-10-19
---

ImageJ2的API文档在[这里](https://javadoc.scijava.org/ImageJ/)。

# ImageJ2的常用Method
基本所有的Method都是继承自SciJava库。
```java
// ImageJ的构造函数，用来创建一个ImageJ的应用上下文
final ImageJ ij = new ImageJ();

// ImageJ2是基于Service的（这与ImageJ1相对）
// plugin service来管理插件
// 获得可用插件的数目
final int pluginCount = ij.plugin().getIndex().size();

// log service来显示日志信息
ij.log().warn("Death Star approaching!");

// status service来报告当前状态
ij.status().showStatus("It's nine o'clock and all is well.");

// menu service来管理菜单层级
final int menuItemCount = ij.menu().getMenu().size();

// platform service来管理与平台相关的功能，比如打开系统的默认浏览器
ij.platform().open(new URL("http://imagej.net/"));

// ui方法来使得用户选择某个文件
final File file = ij.ui().chooseFile(null, "open");

// 加载数据集
final Dataset dataset = ij.scifio().datasetIO().open(file.getPath());

// 运行特定命令
ij.command().run(HelloWorld.class, true);
```

# 打开一张图像
```java
import java.io.File;
import java.io.IOException;

import net.imagej.Dataset;
import net.imagej.ImageJ;

import org.scijava.ItemIO;
import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

import io.scif.services.DatasetIOService;

@Plugin(type = Command.class, menuPath = "Tutorials>Open Image")
public class OpenImage implements Command {
	// 创建数据流服务
	@Parameter
	private DatasetIOService datasetIOService;

	// 创建日志服务，用来捕获异常 
	@Parameter
	private LogService logService;

	// 创建一个基于图形界面的打开文件的方式
	@Parameter
	private File imageFile;

	// 设置输出参数，定义一个Dataset来承载图片
	@Parameter(type = ItemIO.OUTPUT)
	private Dataset image;

	// 运行该命令
	@Override
	public void run() {
		try {
			image = datasetIOService.open(imageFile.getAbsolutePath());
		}
		catch (final IOException exc) {
			// Use the LogService to report the error.
			logService.error(exc);
		}
	}
}
```

# 打开、缩放和存储图片
```java
import io.scif.services.DatasetIOService;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;

import net.imagej.Dataset;
import net.imagej.DefaultDataset;
import net.imagej.ImageJ;
import net.imagej.ImgPlus;
import net.imagej.ops.OpService;
import net.imglib2.RandomAccessible;
import net.imglib2.RandomAccessibleInterval;
import net.imglib2.img.Img;
import net.imglib2.img.ImgView;
import net.imglib2.interpolation.InterpolatorFactory;
import net.imglib2.interpolation.randomaccess.FloorInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.LanczosInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NLinearInterpolatorFactory;
import net.imglib2.interpolation.randomaccess.NearestNeighborInterpolatorFactory;
import net.imglib2.type.numeric.RealType;

import org.scijava.command.Command;
import org.scijava.log.LogService;
import org.scijava.plugin.Parameter;
import org.scijava.plugin.Plugin;

@Plugin(type = Command.class, menuPath = "Tutorials>Open+Scale+Save Image")
public class OpenScaleSaveImage implements Command {

	// 四种缩放方式 
	private static final String LANCZOS = "Lanczos";
	private static final String N_LINEAR = "N-linear";
	private static final String NEAREST_NEIGHBOR = "Nearest neighbor";
	private static final String FLOOR = "Floor";

	// 数据流，用于打开和存储图像
	@Parameter
	private DatasetIOService datasetIOService;

	// 创建OpService，即对图像的操作operation，这里是缩放图像
	@Parameter
	private OpService ops;

	// 创建日志服务
	@Parameter
	private LogService log;

	/** 创建File，用于盛放图片 */
	@Parameter(label = "Image to load")
	private File inputImage;

	/** double型变量，指定缩放因子 */
	@Parameter(label = "Scale factor")
	private double factor = 2;

	// String型变量，指定缩放方法
	@Parameter(label = "Scale method", //
		choices = { LANCZOS, N_LINEAR, NEAREST_NEIGHBOR, FLOOR })
	private String method = LANCZOS;

	/** 创建File型变量，用于盛放存储的图片 */
	@Parameter(label = "Image to save")
	private File outputImage;

	@Override
	public void run() {
		try {
			// 加载图片
			final Dataset image = datasetIOService.open(inputImage.getAbsolutePath());

			// 缩放图片，调用下面的函数
			final Dataset result = scaleImage(image);

			// 存储图片
			datasetIOService.save(result, outputImage.getAbsolutePath());
		}
		catch (final IOException exc) {
			log.error(exc);
		}
	}

	private Dataset scaleImage(final Dataset dataset) {
		// NB: We must do a raw cast through Img, because Dataset does not
		// retain the recursive type parameter; it has an ImgPlus<RealType<?>>.
		// This is invalid for routines that need Img<T extends RealType<T>>.
		@SuppressWarnings({ "rawtypes", "unchecked" })
		final Img<RealType<?>> result = scaleImage((Img) dataset.getImgPlus());

		// Finally, coerce the result back to an ImageJ Dataset object.
		return new DefaultDataset(dataset.context(), new ImgPlus<>(result));
	}
	
	private <T extends RealType<T>> Img<T> scaleImage(final Img<T> image) {
		final double[] scaleFactors = new double[image.numDimensions()];
		Arrays.fill(scaleFactors, factor);
		
		// 根据不同的缩放方式，创建不同的内插函数
		final InterpolatorFactory<T, RandomAccessible<T>> interpolator;
		switch (method) {
			case N_LINEAR:
				interpolator = new NLinearInterpolatorFactory<>();
				break;
			case NEAREST_NEIGHBOR:
				interpolator = new NearestNeighborInterpolatorFactory<>();
				break;
			case LANCZOS:
				interpolator = new LanczosInterpolatorFactory<>();
				break;
			case FLOOR:
				interpolator = new FloorInterpolatorFactory<>();
				break;
			default:
				throw new IllegalArgumentException("Invalid scale method: " + method);
		}

		// 实际调用的是scale函数，用的上面的缩放因子和内插函数
		final RandomAccessibleInterval<T> rai = //
			ops.transform().scale(image, scaleFactors, interpolator);
		return ImgView.wrap(rai, image.factory());
	}

}
```

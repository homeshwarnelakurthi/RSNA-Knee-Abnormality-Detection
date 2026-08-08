"""EDA #3: what is actually inside a DICOM here — tags, pixel data, geometry."""
import pydicom, numpy as np, glob, os

fs = sorted(glob.glob(r"H:\RSNA Knee Abnormality Detection\data\sample_dcm\**\*.dcm", recursive=True))
print("files found:", len(fs))
ds = pydicom.dcmread(fs[0])

print("="*70); print("TRANSFER SYNTAX:", ds.file_meta.TransferSyntaxUID, "|", ds.file_meta.TransferSyntaxUID.name)
print("\nALL RETAINED TAGS:"); print("="*70)
for el in ds:
    if el.tag == (0x7fe0, 0x0010):
        print(f"{el.tag}  {el.name:<38} <PixelData {len(el.value)} bytes>")
    else:
        v = str(el.value)
        print(f"{el.tag}  {el.name:<38} {v[:90]}")

px = ds.pixel_array
print("\n" + "="*70); print("PIXEL DATA")
print("shape:", px.shape, "dtype:", px.dtype)
print("min/max/mean/std:", px.min(), px.max(), round(float(px.mean()),1), round(float(px.std()),1))
print("percentiles 1/50/99:", np.percentile(px,[1,50,99]).round(1))

print("\n" + "="*70); print("GEOMETRY")
for t in ["PixelSpacing","SliceThickness","SpacingBetweenSlices","ImageOrientationPatient",
          "ImagePositionPatient","Rows","Columns","MagneticFieldStrength","Manufacturer",
          "ManufacturerModelName","SeriesDescription","ProtocolName","BodyPartExamined",
          "Laterality","ImageLaterality","PatientSex","PatientAge","RepetitionTime",
          "EchoTime","InversionTime","ScanningSequence","SequenceVariant","ScanOptions",
          "MRAcquisitionType","StudyDescription","PhotometricInterpretation",
          "BitsAllocated","BitsStored","RescaleSlope","RescaleIntercept",
          "WindowCenter","WindowWidth","InstanceNumber","SeriesNumber"]:
    print(f"  {t:<28} {getattr(ds, t, '--ABSENT--')}")

if len(fs) > 1:
    d2 = pydicom.dcmread(fs[1])
    print("\nSecond file IPP:", getattr(d2,"ImagePositionPatient","-"),
          "| InstanceNumber:", getattr(d2,"InstanceNumber","-"))
    print("First  file IPP:", getattr(ds,"ImagePositionPatient","-"),
          "| InstanceNumber:", getattr(ds,"InstanceNumber","-"))

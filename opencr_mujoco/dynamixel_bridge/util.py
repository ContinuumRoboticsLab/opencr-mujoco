import numpy as np

from dynamixel_sdk import DXL_HIBYTE, DXL_LOBYTE, DXL_HIWORD, DXL_LOWORD


def signed_int_to_4_bytes(value):
    if value < 0:
        value = 2**32 + value
    return [
        DXL_LOBYTE(DXL_LOWORD(value)),
        DXL_HIBYTE(DXL_LOWORD(value)),
        DXL_LOBYTE(DXL_HIWORD(value)),
        DXL_HIBYTE(DXL_HIWORD(value)),
    ]


def four_bytes_unsigned_to_signed_int(value):
    if value > 2147483647:
        value = value - 4294967295 - 1
    return value


def mm_to_rad(mm, spool_radii_mm):
    return np.divide(mm, spool_radii_mm)


def rad_to_mm(rad, spool_radii_mm):
    return np.multiply(rad, spool_radii_mm)


def rpm_to_rad_per_sec(rpm):
    return rpm * 2 * np.pi / 60


def rad_per_sec_to_rpm(rad_per_sec):
    return rad_per_sec * 60 / (2 * np.pi)


def get_prepend_string(servo_id: int) -> str:
    return f"[ID: {servo_id:03d}] "

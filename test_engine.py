"""
Test script to check if AttendanceEngine initializes without errors
"""
import sys
import traceback

print("[TEST] Testing AttendanceEngine initialization...")

try:
    from recognition import AttendanceEngine
    print("[TEST] ✓ recognition module imported successfully")
    
    print("[TEST] Creating AttendanceEngine instance...")
    engine = AttendanceEngine()
    print("[TEST] ✓ AttendanceEngine created successfully")
    
    print("[TEST] ✓ All tests passed!")
    
except Exception as e:
    print(f"[TEST] ✗ Error: {type(e).__name__}: {str(e)}")
    traceback.print_exc()
    sys.exit(1)

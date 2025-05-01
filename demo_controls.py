# demo_controls_mouse.py
# A simple script to demonstrate:
# 1) Holding 'w' for 3 seconds (walk forward)
# 2) Smoothly turning the mouse left over 2 seconds using the `mouse` library
# 3) Holding left-click for 7 seconds using the `mouse` library

# IT WOOOOOOORKSSSSS
import pyautogui
import mouse
import time

def smooth_move_with_mouse(dx, dy, duration=1, steps=500):
    """
    Smoothly move the mouse by (dx, dy) over 'duration' seconds,
    splitting the motion into 'steps' small moves.
    """
    delay = duration / steps
    step_x = dx / steps
    step_y = dy / steps
    for _ in range(steps):
        mouse.move(step_x, step_y, absolute=False)
        time.sleep(delay)

def demo():
    # Enable PyAutoGUI failsafe: move mouse to top-left to abort
    pyautogui.FAILSAFE = True

    print("Demo will start in 3 seconds. Move cursor to top-left to abort.")
    time.sleep(3)

    # 1) Hold 'w' for 3 seconds
    '''print("Holding 'w' to walk forward for 3 seconds...")
    pyautogui.keyDown('w')
    time.sleep(3)
    pyautogui.keyUp('w')
    print("Released 'w'.")'''

    # 2) Smooth mouse turn left over 2 seconds
    print("Turning mouse left smoothly for 2 seconds...")
    smooth_move_with_mouse(-500, 0, duration=1)
    print("Mouse turn complete.")

    # 3) Hold left-click for 7 seconds using pyautogui
    '''print("Holding left-click for 7 seconds...")
    pyautogui.mouseDown(button='left')
    time.sleep(7)
    pyautogui.mouseUp(button='left')
    print("Released left-click.")'''

if __name__ == '__main__':
    demo()

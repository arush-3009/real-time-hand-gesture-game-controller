from src.game_control import GameController

print('=' * 60)
print(' ' * 15 + 'HAND GESTURE GAME CONTROL')
print('=' * 60)
print('\nControls:')
print('  • Open hand          → Accelerate (W)')
print('  • Move hand L/R      → Steer (A/D)')
print('  • Make fist          → Brake/Reverse (S)')
print('  • V sign             → Drift (S)')
print('  • Point index finger → Nitro (N)')
print('\nPress "q" to quit')
print('=' * 60)
print()

game_control = GameController()

try:
    game_control.play_game()
except KeyboardInterrupt:
    print('\nInterrupted by user')
except Exception as e:
    print(f'\nError: {e}')
    raise
finally:
    game_control.cleanup()
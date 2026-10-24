# Test servers built on http.server call socket.getfqdn() while binding -- ~35 s each on a Mac CI runner whose name
# resolution hangs. Nothing in the tests needs the answer (set for the whole run: unittest's setUpClass too).
import socket

socket.getfqdn = lambda name="": name or "localhost"

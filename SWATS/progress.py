import sys

#Helper function to show progress during training
def progress(count,total,status=''):
    bar_len=50
    filled_len=int(round(bar_len*count/float(total)))
    percent=round(100.0*count/float(total),1)
    bar='='*filled_len+'-'*(bar_len-filled_len)

    sys.stdout.write('\r[{}] {}{} ...{}\r'.format(bar,percent,'%',status))
    sys.stdout.flush()

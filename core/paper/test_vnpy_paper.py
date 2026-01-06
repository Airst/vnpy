from vnpy.event import EventEngine
from vnpy.trader.engine import MainEngine
from vnpy_paperaccount import PaperAccountApp

def run():
    ee = EventEngine()
    me = MainEngine(ee)
    me.add_app(PaperAccountApp)
    
    pe = me.get_engine("PaperAccount")
    print("Paper Engine Instance:", pe)
    print("Attributes/Methods:", [d for d in dir(pe) if not d.startswith('__')])

    me.close()

if __name__ == "__main__":
    run()

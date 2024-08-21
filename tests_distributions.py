import unittest
import json
import numpy as np
import distributions as dist

def load_data():
    with open('tests/test_cases.json') as f:
        data = json.load(f)
    return data

with open('tests/test_cases.json') as f:
    data = json.load(f)

class ChiSquaredDist_Test(unittest.TestCase):
    
    
    def __init__(self, data):
        self.data = data
        self.dist = dist.ChiSquaredDist(10, 10)

            
    def print_data(self):
        print(json.dumps(self.data, indent=4))
    
    def test_newInstance(self):
        num_cases = len(self.data)
        currnt_case = 0
        print("Number of test cases: ", num_cases)
        count_accurate_calculations = 0
        count_successfull_negative_ = 0
        for d in self.data:
            if num_cases == 0: 
                print("No test cases found")
                break
            else:
                try:
                    self.dist = dist.ChiSquaredDist(d['parameters']['_ν'], d['parameters']['_x'])
                except Exception as e:
                    count_successfull_negative_ += 1
                    pass
                try:
                    self.assertEqual(self.dist._ν, d['parameters']['_ν'], msg = "Line 41 cuaght negative case")
                except Exception as e:
                    count_successfull_negative_ += 1
                    pass
                try:
                    self.assertEqual(self.dist._x, d['parameters']['_x'], msg = "Line 46 caught a negative case")
                except Exception as e:
                    count_successfull_negative_ += 1
                    pass
                print(f"Test case: {d['Case']}")
                print("Testing with x =", d['parameters']["_x"])
                print("Testing with df = ", d['parameters']["_ν"])
                print("Expected pmf: ", d['expected']['pmf'])
                #print("Actual(from my class): ", round(self.dist.pmf(), 4))
                print("Expected: cdf =", d['expected']['cdf'])
                #print("Actual(from my class): ", round(self.dist.cdf(), 4))
                print("\n\n")
                num_cases -= 1
                currnt_case += 1
               
        return count_accurate_calculations, count_successfull_negative_


def main():
    test_instance = ChiSquaredDist_Test(data)
    print(25 * " ~%~")
    print("\n\n\t\t\t\tTesting ChiSquaredDist Class\n\n")
    print(25 * " ~%~")
    print(test_instance.test_newInstance())

if __name__ == "__main__":
    main()